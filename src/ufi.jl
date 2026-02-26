const UFI_POLL_COUNT = Threads.Atomic{Int}(0)

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

# Task Request — MUST match C++ layout
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

    function TaskRequest(work_seq, is_gpu, task_id, args...)
        if task_id == 0
            @error "Task ID 0 is invalid."
        end
        new(work_seq, is_gpu, task_id, args...)
    end
end

struct JuliaTaskRequest # abstracted handle for channel push! and take!
    req::TaskRequest
    args::Vector{TaskArgument}
    slot_id::Int
end


const TASK_REGISTRY = Dict{UInt32,Union{CPUWrapType,Function}}()
const REGISTRY_LOCK = ReentrantLock()
const LAST_CREATED_TASK_ID = Threads.Atomic{Int}(0)

const UFI_INIT_LOCK = ReentrantLock()

const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)
const MAX_SUBMITTED_TASK_ID = Threads.Atomic{Int}(0)

const MAX_UFI_SLOTS = Ref{Int}(0)
const SLOT_REQUEST_PTRS = Vector{Ptr{TaskRequest}}()
const UFI_LAST_TASK_IDS = Vector{Threads.Atomic{Int32}}()
const UFI_WORKER_TASKS = Vector{Task}()


const JOB_QUEUE = Ref{Channel{JuliaTaskRequest}}()

const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)
const UFI_EXEC_LOCK = Ref{ReentrantLock}()

const UFI_POLLER_TIMER = Ref{Base.Timer}()
const UFI_POLL_INTERVAL = 0.001 # 1ms

function get_pending_tasks()
    return ccall((:legate_get_pending_tasks_count, Legate.WRAPPER_LIB_PATH), Cint, ())
end

function get_active_call_count()
    return ccall((:legate_get_active_call_count, Legate.WRAPPER_LIB_PATH), Cint, ())
end

function get_active_slot_count()
    return ccall((:legate_get_active_slot_count, Legate.WRAPPER_LIB_PATH), Cint, ())
end

function get_max_task_id_seen()
    return ccall((:legate_get_max_task_id_seen, Legate.WRAPPER_LIB_PATH), Cint, ())
end

function ufi_has_pending_work()
    submitted = MAX_SUBMITTED_TASK_ID[]
    seen = Int(get_max_task_id_seen())
    return submitted > seen || get_active_call_count() > 0 || get_active_slot_count() > 0
end

const UFI_POLL_LOCK = ReentrantLock()

function wait_ufi()
    @debug "Waiting for UFI to complete"
    Legate.issue_execution_fence(blocking=false)

    while ufi_has_pending_work()
        ufi_poll_sync() # Manual Poll: Drive progress on main thread
        sleep(0.001) 
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
end

function create_julia_task(rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaTask)
    if task_obj isa JuliaCPUTask
        id = LegateInternal.JULIA_CUSTOM_TASK
        task_impl = LegateInternal.create_auto_task(rt, lib, id)
    else
        id = LegateInternal.JULIA_CUSTOM_GPU_TASK
        task_impl = LegateInternal.create_auto_task(rt, lib, id)
    end
    
    task = AutoTask(task_impl)
    task.task_id = task_obj.task_id
    
    add_scalar(task, Scalar(Int32(task_obj.task_id)))
    register_task_function(task_obj.task_id, task_obj.fun)
    Threads.atomic_xchg!(LAST_CREATED_TASK_ID, Int(task_obj.task_id))
    return task
end

function _completion_callback(slot_id::Int)
    ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), slot_id)
end

function ufi_poll_sync()
    tid = Threads.threadid()
    if !(tid == 1)
        return
    end
    
    try
        if UFI_SHUTDOWN[]
            return
        end

        # 2. Poll Slots
        # Popping from the C++ pending queue to avoid O(N) scanning
        while true
            found_slot = LegateInternal.legate_pop_pending_slot()
            if found_slot == -1
                break
            end
            
            i = found_slot + 1 # Julia is 1-indexed
            req = unsafe_load(SLOT_REQUEST_PTRS[i])
            dims = ntuple(i -> Int(max(0, req.dims[i])), req.ndim)
            args = Vector{TaskArgument}(undef, Int(req.num_inputs + req.num_outputs + req.num_scalars))
            _fill_args_core!(args, req, dims)

            @debug "UFI: Found pending slot" thread=tid slot_id=found_slot req=req args=args
            put!(JOB_QUEUE[], JuliaTaskRequest(req, args, found_slot))
            return
        end
    catch e
        @error "UFI: Critical error in ufi_poll_sync" thread=Threads.threadid() exception=(e, catch_backtrace())
        return
    end
end


function _start_ufi_system()
    UFI_SHUTDOWN[] = false
end

function _execute_julia_task_internal(job::JuliaTaskRequest)
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[job.req.task_id]
    end
    
    if job.req.is_gpu != 0
        error("Legate UFI: GPU execution not supported.")
    else
        # task_fun(job.args...)
        Base.invokelatest(task_fun, job.args...)
        _completion_callback(job.slot_id)
    end
end

function _ufi_worker_loop()
    while !UFI_SHUTDOWN[]
        try
            # take! blocks until a job is available
            job = take!(JOB_QUEUE[])
            @debug "UFI: executing task" slot_id=job.slot_id req=job.req
            
            _execute_julia_task_internal(job)
        catch e
            if !UFI_SHUTDOWN[]
                @error "UFI Worker: Unexpected error" exception=(e, catch_backtrace())
            end
        end
    end
end

# Field access helpers for TaskRequest
_get_type(x::Ptr{Cint}, i) = unsafe_load(x, i)
_get_ptr(x::Ptr{Ptr{Cvoid}}, i) = unsafe_load(x, i)

function _fill_args_core!(args, req, dims)
    offset = 1
    for i in 1:req.num_inputs
        T = get_code_type(_get_type(req.inputs_types, i))
        ptr = Ptr{T}(_get_ptr(req.inputs_ptr, i))
        args[offset] = unsafe_wrap(Array, ptr, dims)
        offset += 1
    end
    for i in 1:req.num_outputs
        T = get_code_type(_get_type(req.outputs_types, i))
        ptr = Ptr{T}(_get_ptr(req.outputs_ptr, i))
        args[offset] = unsafe_wrap(Array, ptr, dims)
        offset += 1
    end

    for i in 1:req.num_scalars
        T = get_code_type(Int(_get_type(req.scalar_types, i)))
        val_ptr = _get_ptr(req.scalars_ptr, i)
        args[offset] = unsafe_load(Ptr{T}(val_ptr))
        offset += 1
    end
end

function init_ufi()
    lock(UFI_INIT_LOCK) do
        UFI_INITIALIZED[] && return
        _is_precompiling() && return
        
        JOB_QUEUE[] = Channel{JuliaTaskRequest}(10000)

        UFI_EXEC_LOCK[] = ReentrantLock()

        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        MAX_UFI_SLOTS[] = max_slots
        
        resize!(UFI_LAST_TASK_IDS, max_slots)
        for i in 1:max_slots
            UFI_LAST_TASK_IDS[i] = Threads.Atomic{Int32}(0)
        end
        
        empty!(SLOT_REQUEST_PTRS)
        for i in 1:max_slots
            req_ptr = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{TaskRequest}, (Cint,), i-1)
            push!(SLOT_REQUEST_PTRS, req_ptr)
        end

        LegateInternal._initialize_async_system()

        # Precompile callbacks on main thread to avoid JIT on worker threads
        try
            precompile(ufi_poll_sync, ())
            precompile(_completion_callback, (Int,))
            precompile(_execute_julia_task_internal, (JuliaTaskRequest,))
            precompile(_fill_args_core!, (Vector{TaskArgument}, TaskRequest, NTuple{3, Int64}))
        catch e
            @warn "Precompilation failed" exception=(e, catch_backtrace())
        end
        UFI_INITIALIZED[] = true
    end

    _start_ufi_threads()
    
    # Spawn dedicated worker threads on background threads if available
    n = Threads.nthreads()
    num_workers = max(1, n - 1)
    empty!(UFI_WORKER_TASKS)
    
    if n > 1
        for i in 2:n
            push!(UFI_WORKER_TASKS, Threads.@spawn _ufi_worker_loop())
        end
    else
        # Single-threaded fallback
        push!(UFI_WORKER_TASKS, Threads.@spawn _ufi_worker_loop())
    end
end

function _start_ufi_threads()
    UFI_INITIALIZED[] && isassigned(UFI_POLLER_TIMER) && return

    @debug "UFI System: Initializing Main-Thread Poller (Interval: $(UFI_POLL_INTERVAL)s)"
    UFI_POLLER_TIMER[] = Base.Timer(0.0; interval=UFI_POLL_INTERVAL) do timer
        if !UFI_SHUTDOWN[]
            ufi_poll_sync()
        else
            close(timer)
        end
    end
end

function shutdown_ufi()
    UFI_SHUTDOWN_DONE[] && return nothing
    UFI_SHUTDOWN[] = true

    if isassigned(JOB_QUEUE) && isopen(JOB_QUEUE[])
        close(JOB_QUEUE[])
    end
    for task in UFI_WORKER_TASKS
        try fetch(task) catch end
    end
    empty!(UFI_WORKER_TASKS)
    UFI_SHUTDOWN_DONE[] = true
end