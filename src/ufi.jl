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

const TaskArgument = Union{AbstractArray,SUPPORTED_TYPES}
const CPUWrapType = Function

struct JuliaCPUTask
    fun::CPUWrapType
    task_id::UInt32
end

struct JuliaGPUTask
    fun::Function
    task_id::UInt32
end

JuliaTask = Union{JuliaCPUTask,JuliaGPUTask}

# Task Request structure - MUST match C++ layout
struct TaskRequest
    work_seq::UInt64
    is_gpu::Cint
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
        new(0, 0, 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0, 0, (0, 0, 0))
    end
end

const TASK_REGISTRY = Dict{UInt32,Union{CPUWrapType,Function}}()
const REGISTRY_LOCK = ReentrantLock()

const UFI_INIT_LOCK = ReentrantLock()

const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)
const PENDING_TASKS = Threads.Atomic{Int}(0)
const COMPLETED_TASKS = Threads.Atomic{Int}(0)

const MAX_UFI_SLOTS = Ref{Int}(0)
const UFI_ASYNC_COND = Ref{Base.AsyncCondition}()
const SLOT_WORK_AVAILABLE_PTRS = Vector{Ptr{Int32}}()
const SLOT_REQUEST_PTRS = Vector{Ptr{TaskRequest}}()
const UFI_LAST_TASK_IDS = Vector{UInt64}()
const UFI_WORKER_TASKS = Vector{Task}()
const JOB_QUEUE = Ref{Channel{Pair{Int,TaskRequest}}}()
const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)

function get_pending_tasks()
    return ccall((:legate_get_pending_tasks_count, Legate.WRAPPER_LIB_PATH), Cint, ())
end

function ufi_has_pending_work()
    return PENDING_TASKS[] > 0 || get_pending_tasks() > 0
end

function wait_ufi()
    while ufi_has_pending_work()
        sleep(0.01)
        yield()
    end
end

function wrap_task(f; task_type=:cpu)
    task_id = Threads.atomic_add!(NEXT_TASK_ID, UInt32(1))
    if task_type == :gpu
        return JuliaGPUTask(f, task_id)
    else
        return JuliaCPUTask(f, task_id)
    end
end

function register_task_function(id::UInt32, fun::Union{CPUWrapType,Function})
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[id] = fun
    end
    precompile(fun, (Vector{Any},))
    precompile(Base.invokelatest, (typeof(fun), Vector{Any}))
end

function create_julia_task(rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaCPUTask)
    task = create_task(rt, lib, JULIA_CUSTOM_TASK)
    add_scalar(task, Scalar(task_obj.task_id))
    register_task_function(task_obj.task_id, task_obj.fun)
    return task
end

function _completion_callback(slot_id::Int)
    ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), slot_id)
end

function ufi_poller_task()
    tid = Threads.threadid()
    @debug "ufi_poller_task starting" thread=tid
    try
        while !UFI_SHUTDOWN[]
            num_slots = length(SLOT_WORK_AVAILABLE_PTRS)
            work_found = false
            for i in 1:num_slots
                val = UInt64(unsafe_load(SLOT_WORK_AVAILABLE_PTRS[i]))
                if val != 0 && val != UFI_LAST_TASK_IDS[i]
                    Threads.atomic_fence()
                    req_ptr = SLOT_REQUEST_PTRS[i]
                    while unsafe_load(Ptr{UInt64}(req_ptr)) != val
                        yield()
                    end
                    req_copy = unsafe_load(req_ptr)
                    UFI_LAST_TASK_IDS[i] = val
                    Threads.atomic_add!(PENDING_TASKS, 1)
                    put!(JOB_QUEUE[], (i - 1) => req_copy)
                    work_found = true
                end
            end
            if !work_found
                try wait(UFI_ASYNC_COND[]) catch; break; end
            else
                yield()
            end
        end
    catch e
        @error "UFI Poller error" exception=(e, catch_backtrace())
    end
    @debug "ufi_poller_task finished" thread=tid
end

function ufi_worker_task()
    tid = Threads.threadid()
    @debug "ufi_worker_task starting" thread=tid

    args = Vector{TaskArgument}()
    sizehint!(args, 64)
    try
        for (slot_id, req) in JOB_QUEUE[]
            try
                execute_julia_task(req, slot_id, args)
            catch e
                @error "UFI Worker task failure" exception=(e, catch_backtrace())
            finally
                _completion_callback(slot_id)
                Threads.atomic_add!(COMPLETED_TASKS, 1)
                Threads.atomic_sub!(PENDING_TASKS, 1)
            end
        end
    catch e
        if !UFI_SHUTDOWN[]
            @error "UFI Worker error" exception=(e, catch_backtrace())
        end
    end
    @debug "ufi_worker_task finished" thread=tid
end

function execute_julia_task(req::TaskRequest, slot_id::Integer, args::Vector{TaskArgument})
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[req.task_id]
    end
    
    if req.is_gpu != 0
        error("Legate UFI: GPU execution not supported.")
    else
        _execute_julia_task_cpu(req, task_fun, args)
    end
end

function _execute_julia_task_cpu(req::TaskRequest, task_fun::CPUWrapType, args::Vector{TaskArgument})
    empty!(args)
    dims = ntuple(i -> max(0, req.dims[i]), req.ndim)
    
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

    @debug "UFI $task_fun Task executed"
    Base.invokelatest(task_fun, args)
end

function execute_julia_task(req::TaskRequest, slot_id::Integer)
    task_id = req.task_id
    
    # 1. Thread-safe lookup of task function
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[task_id]
    end

    # Dispatch task
    _dispatch_task(req, task_fun)
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

function _start_ufi_system()
    UFI_SHUTDOWN[] = false
    JOB_QUEUE[] = Channel{Pair{Int,TaskRequest}}(MAX_UFI_SLOTS[])
    empty!(UFI_WORKER_TASKS)

    @debug "_start_ufi_system: Booting true concurrent UFI pool"

    # Poller runs independently on main thread pool
    push!(UFI_WORKER_TASKS, @async ufi_poller_task())
    sleep(0.1)

    nworkers = 4
    for i in 1:nworkers
        @debug "_start_ufi_system: Spawning worker $i"
        push!(UFI_WORKER_TASKS, Threads.@spawn ufi_worker_task())
        sleep(0.1)
    end
    yield()
end

# --- Initialization ---

function init_ufi()
    lock(UFI_INIT_LOCK) do
        UFI_INITIALIZED[] && return
        _is_precompiling() && return
        
        UFI_ASYNC_COND[] = Base.AsyncCondition()
        ccall((:legate_set_async_handle, Legate.WRAPPER_LIB_PATH), Cvoid, (Ptr{Cvoid},), UFI_ASYNC_COND[].handle)

        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        MAX_UFI_SLOTS[] = max_slots
        
        @debug "init_ufi" max_slots sizeof_TR=sizeof(TaskRequest)
        
        resize!(UFI_LAST_TASK_IDS, max_slots)
        fill!(UFI_LAST_TASK_IDS, 0)
        
        empty!(SLOT_WORK_AVAILABLE_PTRS)
        empty!(SLOT_REQUEST_PTRS)
        for i in 1:max_slots
            push!(SLOT_WORK_AVAILABLE_PTRS, ccall((:legate_get_slot_work_available_ptr, Legate.WRAPPER_LIB_PATH), Ptr{Int32}, (Cint,), i-1))
            push!(SLOT_REQUEST_PTRS, ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{TaskRequest}, (Cint,), i-1))
        end

        _start_ufi_system()
        UFI_INITIALIZED[] = true
    end
end

function shutdown_ufi()
    UFI_SHUTDOWN_DONE[] && return nothing
    UFI_SHUTDOWN[] = true
    if isassigned(UFI_ASYNC_COND)
        try close(UFI_ASYNC_COND[]) catch end
    end
    if isassigned(JOB_QUEUE) && isopen(JOB_QUEUE[])
        close(JOB_QUEUE[])
    end
    for task in UFI_WORKER_TASKS
        try fetch(task) catch end
    end
    empty!(UFI_WORKER_TASKS)
    UFI_SHUTDOWN_DONE[] = true
end