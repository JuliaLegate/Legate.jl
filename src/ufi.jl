#= Copyright 2026 Northwestern University, 
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 =#

export JuliaGPUTask, JuliaCPUTask, JuliaTask, TaskArgument, TaskRequest

# Legate scalar types + AbstractArray for tasks
const TaskArgument = Union{AbstractArray,SUPPORTED_TYPES}

const CPUWrapType = FunctionWrapper{Nothing,Tuple{Vector{TaskArgument}}}

struct JuliaCPUTask
    fun::CPUWrapType
    task_id::UInt32
end

struct JuliaGPUTask
    fun::Function
    task_id::UInt32
end

JuliaTask = Union{JuliaCPUTask,JuliaGPUTask}

function wrap_task(f; task_type=:cpu)
    task_id = Threads.atomic_add!(NEXT_TASK_ID, UInt32(1))
    if task_type == :gpu
        return JuliaGPUTask(f, task_id)
    else
        return JuliaCPUTask(CPUWrapType(f), task_id)
    end
end

# Thread-safe execution from Legate worker threads
# Signals via uv_async_send, Julia executes

# Shared data structure for passing task information from C++ to Julia
struct TaskRequest
    is_gpu::Cint     # Use Cint for better alignment
    task_id::UInt32
    inputs_ptr::Ptr{Ptr{Cvoid}}
    outputs_ptr::Ptr{Ptr{Cvoid}}
    scalars_ptr::Ptr{Ptr{Cvoid}}
    inputs_types::Ptr{Cint}
    outputs_types::Ptr{Cint}
    scalar_types::Ptr{Cint}
    num_inputs::Csize_t
    num_outputs::Csize_t
    num_scalars::Csize_t
    ndim::Cint
    dims::NTuple{3,Int64}

    function TaskRequest()
        new(0, 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0, 0, (0, 0, 0))
    end
end

# Union{CPUWrapType,Function} to allow storing both CPU FunctionWrappers and GPU kernel functions
const TASK_REGISTRY = Dict{UInt32,Union{CPUWrapType,Function}}()
const REGISTRY_LOCK = ReentrantLock()

# Atomic counter for auto-generating task IDs
const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)

# Task Synchronization
const PENDING_TASKS = Threads.Atomic{Int}(0)
const COMPLETED_TASKS = Threads.Atomic{Int}(0)

# Track sequence IDs to avoid duplicate queuing in Poller
const UFI_LAST_TASK_IDS = Vector{Int32}() # ACTUALLY SEQUENCE IDs NOW

# Track ALL spawned tasks for clean joining at shutdown
const UFI_ACTIVE_TASKS = Set{Task}()
const UFI_TASKS_LOCK = ReentrantLock()
const ALL_TASKS_DONE = Threads.Condition()
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)

get_pending_tasks() = ccall((:legate_get_pending_tasks_count, Legate.WRAPPER_LIB_PATH), Cint, ())

function ufi_has_pending_work()
    # Check execution counter
    if PENDING_TASKS[] > 0
        return true
    end
    # Check global C++ active task counter
    active_cpp = get_pending_tasks()
    return active_cpp > 0
end

function wait_ufi()
    while true
        pending_julia = PENDING_TASKS[]
        active_cpp = get_pending_tasks()
        
        if pending_julia <= 0 && active_cpp <= 0
            # Double check to handle submission lag
            sleep(0.5)
            pending_julia = PENDING_TASKS[]
            active_cpp = get_pending_tasks()
            if pending_julia <= 0 && active_cpp <= 0
                break
            end
        end
        
        yield()
        sleep(0.1)
    end
end

function register_task_function(id::UInt32, fun::Union{CPUWrapType,Function})
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[id] = fun
    end
end

@doc"""
    create_julia_task(rt::Runtime, lib::Library, task_obj::JuliaTask) -> AutoTask

Create a Julia task in the runtime.

# Arguments
- `rt`: The current runtime instance.
- `lib`: The library to associate with the task.
- `task_obj`: The Julia task object to register.
"""
function create_julia_task(
    rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaCPUTask
)
    task = create_task(rt, lib, JULIA_CUSTOM_TASK)
    add_scalar(task, Scalar(task_obj.task_id))
    register_task_function(task_obj.task_id, task_obj.fun)
    return task
end

# in CUDAExt ufi.jl
# function create_julia_task(
#     rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaGPUTask
# ) end

const SLOT_WORK_AVAILABLE_PTRS = Vector{Ptr{Int32}}()

# Global state for multi-threaded UFI
const MAX_UFI_SLOTS = 8
const SLOT_REQUEST_PTRS = Vector{Ptr{TaskRequest}}()
const UFI_CHANNEL = Channel{Int}(MAX_UFI_SLOTS)
const UFI_ASYNC_COND = Ref{Base.AsyncCondition}()

const UFI_WORKER_TASKS = Vector{Task}()
const JOB_QUEUE = Ref{Channel{Int}}()
const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)

# Dedicated wrappers for task claiming and completion
# to ensure ccall sites are warmed up on main thread
function _try_claim_slot(slot_id::Int)
    return ccall((:legate_try_claim_slot, Legate.WRAPPER_LIB_PATH), Bool, (Cint,), slot_id)
end

function _completion_callback(slot_id::Int)
    ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), slot_id)
end

# Receives signals from C++ or polls, and executes tasks
function ufi_poller_task()
    @info "ufi_poller_task starting" thread=Threads.threadid()
    try
        while !UFI_SHUTDOWN[]
            # 1. DRAIN all slots
            num_slots = length(SLOT_WORK_AVAILABLE_PTRS)
            work_found = false
            for i in 1:num_slots
                 val = unsafe_load(SLOT_WORK_AVAILABLE_PTRS[i])
                 if val != 0
                      Threads.atomic_fence()
                      if val != UFI_LAST_TASK_IDS[i]
                          @debug "Poller: Found work in slot $(i-1) (seq=$val)"
                          UFI_LAST_TASK_IDS[i] = val
                          Threads.atomic_add!(PENDING_TASKS, 1)
                          put!(JOB_QUEUE[], i - 1)
                          work_found = true
                      end
                 end
            end
            
            if !work_found
                sleep(0.01)
            else
                yield()
            end
        end
    catch e
        @error "Poller CRASHED: $e"
    end
    @debug "Poller exiting."
end

# Worker Task
function ufi_worker_task()
    tid = Threads.threadid()
    @info "ufi_worker_task starting" thread=tid
    try
        for slot_id in JOB_QUEUE[]
            @debug "Worker($tid): Picking up slot $slot_id"
            req = unsafe_load(SLOT_REQUEST_PTRS[slot_id + 1])
            try
                execute_julia_task(req, slot_id)
            catch e
                @debug "Worker($tid): Task failed in slot $slot_id: $e"
            finally
                _completion_callback(slot_id)
                Threads.atomic_add!(Legate.COMPLETED_TASKS, 1)
                Threads.atomic_sub!(PENDING_TASKS, 1)
            end
        end
    catch e
        @error "Worker($tid) CRASHED: $e"
    end
    @debug "Worker($tid) exiting."
end


# in CUDAExt ufi.jl
# GPU execution stub (overridden by CUDA extension)
function _execute_julia_task_gpu(req, task_fun)
    error("Legate UFI: GPU task execution not implemented or CUDA extension not loaded.")
end

function _execute_julia_task_cpu(req, task_fun)
    args = Vector{TaskArgument}()
    sizehint!(args, Int(req.num_inputs + req.num_outputs + req.num_scalars))

    dims = ntuple(i -> req.dims[i], Int(req.ndim))
    for i in 1:req.num_inputs
        type_code = unsafe_load(req.inputs_types, i)
        T = get_code_type(type_code)
        ptr = Ptr{T}(unsafe_load(req.inputs_ptr, i))
        push!(args, unsafe_wrap(Array, ptr, dims))
    end

    for i in 1:req.num_outputs
        type_code = unsafe_load(req.outputs_types, i)
        T = get_code_type(type_code)
        ptr = Ptr{T}(unsafe_load(req.outputs_ptr, i))
        push!(args, unsafe_wrap(Array, ptr, dims))
    end

    for i in 1:req.num_scalars
        type_code = Int(unsafe_load(req.scalar_types, i))
        T = get_code_type(type_code)
        val_ptr = unsafe_load(req.scalars_ptr, i)
        val = unsafe_load(Ptr{T}(val_ptr))
        push!(args, val)
    end

    @debug "UFI Task executed in slot task_fun=$task_fun args=$args"
    task_fun(args)
end

function _dispatch_task(req::TaskRequest, task_fun)
    if req.is_gpu != 0
        # GPU Dispatch
        _execute_julia_task_gpu(req, task_fun)
    else
        # CPU Dispatch
        _execute_julia_task_cpu(req, task_fun)
    end
end


function execute_julia_task(req::TaskRequest, slot_id::Integer)
    task_id = req.task_id
    
    # Looking up task function by ID (thread-safe via REGISTRY_LOCK)
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[task_id]
    end

    try
        _dispatch_task(req, task_fun)
    catch e
        @error "Legate UFI: task_fun FAILED" task_id=task_id exception=(e, catch_backtrace())
        rethrow(e)
    finally
        # task is done
    end
end

function _start_ufi_system()
    # Initialize System state
    UFI_SHUTDOWN[] = false
    JOB_QUEUE[] = Channel{Int}(MAX_UFI_SLOTS)
    empty!(UFI_WORKER_TASKS)

    # Spawn the Poller on default pool
    push!(UFI_WORKER_TASKS, Threads.@spawn ufi_poller_task())

    nworkers = 4
    for i in 1:nworkers
        push!(UFI_WORKER_TASKS, Threads.@spawn ufi_worker_task())
        sleep(0.1)
    end

    sleep(0.1)
    yield()
end

# Initialize and start worker on INTERACTIVE thread loop
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_INIT_LOCK = ReentrantLock()

function init_ufi()
    lock(UFI_INIT_LOCK) do
        UFI_INITIALIZED[] && return
        _is_precompiling() && return
    UFI_ASYNC_COND[] = Base.AsyncCondition()
    
    ccall((:legate_set_async_handle, Legate.WRAPPER_LIB_PATH), Cvoid, (Ptr{Cvoid},), UFI_ASYNC_COND[].handle)

    max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
    @info "init_ufi" max_slots sizeof_TaskRequest=sizeof(TaskRequest)

    resize!(SLOT_REQUEST_PTRS, max_slots)
    resize!(SLOT_WORK_AVAILABLE_PTRS, max_slots)

    for i in 1:max_slots
        SLOT_REQUEST_PTRS[i] = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{TaskRequest}, (Cint,), i-1)
        SLOT_WORK_AVAILABLE_PTRS[i] = ccall((:legate_get_slot_work_available_ptr, Legate.WRAPPER_LIB_PATH), Ptr{Int32}, (Cint,), i-1)
    end

    empty!(UFI_LAST_TASK_IDS)
    resize!(UFI_LAST_TASK_IDS, max_slots)

    fill!(UFI_LAST_TASK_IDS, Int32(0))
    _start_ufi_system()
    UFI_INITIALIZED[] = true
    end
end

function shutdown_ufi()
    # Prevent double shutdown
    UFI_SHUTDOWN_DONE[] && return nothing
    
    # Wait for all pending tasks to drain before signaling shutdown
    wait_ufi()
    
    # Signal poller and workers to stop
    UFI_SHUTDOWN[] = true
    if isassigned(JOB_QUEUE) && isopen(JOB_QUEUE[])
        close(JOB_QUEUE[])
    end
    
    # Wait for poller and workers to exit
    for task in UFI_WORKER_TASKS
        try
            wait(task)
        catch
        end
    end
    empty!(UFI_WORKER_TASKS)
    
    while true
        tasks_to_wait = lock(UFI_TASKS_LOCK) do
            collect(UFI_ACTIVE_TASKS)
        end
        
        isempty(tasks_to_wait) && break
        
        # Join the tasks
        for t in tasks_to_wait
            try
                wait(t)
            catch
            end
        end
        yield()
    end

    if isassigned(UFI_ASYNC_COND)
        try
            close(UFI_ASYNC_COND[])
        catch
        end
    end
    
    UFI_SHUTDOWN_DONE[] = true
end