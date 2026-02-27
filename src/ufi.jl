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

using StaticArrays

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

# Task Request — MUST match C++ layout in task.cpp
struct TaskRequest
    is_gpu::Cint        # Offset 0
    task_id::UInt32     # Offset 4
    inputs_ptr::Ptr{Ptr{Cvoid}} # Offset 8
    outputs_ptr::Ptr{Ptr{Cvoid}} # Offset 16
    scalars_ptr::Ptr{Ptr{Cvoid}} # Offset 24
    inputs_types::Ptr{Cint} # Offset 32
    outputs_types::Ptr{Cint} # Offset 40
    scalar_types::Ptr{Cint} # Offset 48
    num_inputs::Csize_t # Offset 56
    num_outputs::Csize_t # Offset 64
    num_scalars::Csize_t # Offset 72
    ndim::Cint          # Offset 80
    dims::NTuple{3,Int64} # Offset 88

    function TaskRequest(is_gpu, task_id, args...)
        if task_id == 0
            @error "Task ID 0 is invalid."
        end
        new(is_gpu, task_id, args...)
    end
end

# Monomorphic struct for arguments to prevent JIT contention in workers.
# Using NTuple{16, Any} allows us to avoid parametric JuliaTaskRequest entirely.
struct JuliaTaskRequest
    task_fun::CPUWrapType
    args::NTuple{16, Any}
    num_args::Int
    slot_id::Int
end

const UFI_ERROR_CODE = 123

const TASK_REGISTRY = Dict{UInt32,Union{CPUWrapType,Function}}()
const REGISTRY_LOCK = ReentrantLock()

const UFI_INIT_LOCK = ReentrantLock()

const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)
const MAX_SUBMITTED_TASK_ID = Threads.Atomic{Int}(0)

const MAX_UFI_SLOTS_VAL = 32
const MAX_UFI_SLOTS = Ref{Int}(MAX_UFI_SLOTS_VAL)
# StaticArrays provide a cleaner high-level interface while keeping NTuple performance
const SLOT_REQUEST_PTRS = Ref{SVector{MAX_UFI_SLOTS_VAL, Ptr{TaskRequest}}}()
const UFI_WORKER_TASKS = Vector{Task}(undef, 64)
const UFI_WORKER_COUNT = Threads.Atomic{Int}(0)

const JOB_QUEUE = Ref{Channel{JuliaTaskRequest}}()

const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)

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
    return submitted > seen || isready(JOB_QUEUE[]) || get_active_call_count() > 0 || get_active_slot_count() > 0
end

function wait_ufi()
    @debug "Waiting for UFI to complete"
    Legate.issue_execution_fence(blocking=false)

    while ufi_has_pending_work()
        ufi_poll_sync() # Manual Poll: Drive progress on main thread
        yield() # Yield for lower latency and better responsiveness
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
    task.fun = task_obj.fun

    add_scalar(task, Scalar(Int32(task_obj.task_id)))
    register_task_function(task_obj.task_id, task_obj.fun)
    return task
end

function _completion_callback(slot_id::Int)
    ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), slot_id)
end

# Helper to get the i-th argument from TaskRequest as a concrete Julia type.
@inline function _get_arg(req, i, dims)
    if i <= req.num_inputs # Inputs
        T = get_code_type(_get_type(req.inputs_types, i))
        ptr_val = _get_ptr(req.inputs_ptr, i)
        ptr_val == C_NULL && error("UFI: Input pointer is NULL for arg $i")
        return unsafe_wrap(Array, Ptr{T}(ptr_val), dims)
    elseif i <= req.num_inputs + req.num_outputs # Outputs
        idx = i - req.num_inputs
        T = get_code_type(_get_type(req.outputs_types, idx))
        ptr_val = _get_ptr(req.outputs_ptr, idx)
        ptr_val == C_NULL && error("UFI: Output pointer is NULL for arg $i")
        return unsafe_wrap(Array, Ptr{T}(ptr_val), dims)
    else # Scalars
        idx = Int(i - req.num_inputs - req.num_outputs)
        T = get_code_type(Int(_get_type(req.scalar_types, idx)))
        val_ptr = _get_ptr(req.scalars_ptr, idx)
        val_ptr == C_NULL && error("UFI: Scalar pointer is NULL for arg $i")
        return unsafe_load(Ptr{T}(val_ptr))
    end
end

const IN_POLL = Threads.Atomic{Bool}(false)
function ufi_poll_sync()
    if Threads.threadid() != 1
        return
    end
    # Atomic re-entrancy guard: if another task yielded while in poll, skip.
    if Threads.atomic_cas!(IN_POLL, false, true)
        return
    end
    try
        if UFI_SHUTDOWN[]
            return
        end

        while true
            fs = Int(LegateInternal.legate_pop_pending_slot())
            fs == -1 && break

            req = unsafe_load(SLOT_REQUEST_PTRS[][fs + 1])
            dims = ntuple(i -> Int(max(0, req.dims[i])), req.ndim)
            
            task_fun = lock(REGISTRY_LOCK) do
                TASK_REGISTRY[req.task_id]
            end

            num_args = Int(req.num_inputs + req.num_outputs + req.num_scalars)
            args = ntuple(i -> (i <= num_args ? _get_arg(req, i, dims) : nothing), Val(16))
            
            job = JuliaTaskRequest(task_fun, args, num_args, fs)
            put!(JOB_QUEUE[], job)
        end
    catch e
        @error "UFI: FATAL error in ufi_poll_sync" thread=Threads.threadid() exception=(e, catch_backtrace())
        exit(UFI_ERROR_CODE) # Fail Fast: Exit on any UFI logic corruption
    finally
        IN_POLL[] = false
    end
end


function _start_ufi_system()
    UFI_SHUTDOWN[] = false
end

function _execute_julia_task_internal(@nospecialize(job::JuliaTaskRequest))
    # Nospecialize + Manual Branches = Zero JIT churn in workers.
    try
        n = job.num_args
        args = job.args
        if n == 0
            Base.invokelatest(job.task_fun)
        elseif n == 1
            Base.invokelatest(job.task_fun, args[1])
        elseif n == 2
            Base.invokelatest(job.task_fun, args[1], args[2])
        elseif n == 3
            Base.invokelatest(job.task_fun, args[1], args[2], args[3])
        elseif n == 4
            Base.invokelatest(job.task_fun, args[1], args[2], args[3], args[4])
        elseif n == 5
            Base.invokelatest(job.task_fun, args[1], args[2], args[3], args[4], args[5])
        elseif n == 6
            Base.invokelatest(job.task_fun, args[1], args[2], args[3], args[4], args[5], args[6])
        elseif n == 7
            Base.invokelatest(job.task_fun, args[1], args[2], args[3], args[4], args[5], args[6], args[7])
        elseif n == 8
            Base.invokelatest(job.task_fun, args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
        else
            Base.invokelatest(job.task_fun, args[1:n]...)
        end
    catch e
        @error "UFI Execution Error" slot_id=job.slot_id exception=(e, catch_backtrace())
        exit(UFI_ERROR_CODE) # Fail Fast: Any worker corruption is fatal
    finally
        _completion_callback(job.slot_id)
    end
end

function _ufi_worker_loop()
    while !UFI_SHUTDOWN[]
        try
            # take! blocks until a job is available
            job = take!(JOB_QUEUE[])
            _execute_julia_task_internal(job)
        catch e
            if !UFI_SHUTDOWN[]
                @error "UFI Worker: FATAL error" exception=(e, catch_backtrace())
                exit(UFI_ERROR_CODE) # Fail Fast: Any worker corruption is fatal
            end
        end
    end
end

# Field access helpers for TaskRequest
_get_type(x::Ptr{Cint}, i) = unsafe_load(x, i)
_get_ptr(x::Ptr{Ptr{Cvoid}}, i) = unsafe_load(x, i)

function init_ufi()
    lock(UFI_INIT_LOCK) do
        UFI_INITIALIZED[] && return
        _is_precompiling() && return
        
        JOB_QUEUE[] = Channel{JuliaTaskRequest}(128)

        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        MAX_UFI_SLOTS[] = max_slots
        
        # Initialize SLOT_REQUEST_PTRS as a fixed-size StaticArray
        tmp_ptrs = Vector{Ptr{TaskRequest}}(undef, MAX_UFI_SLOTS_VAL)
        fill!(tmp_ptrs, Ptr{TaskRequest}(C_NULL))
        for i in 1:max_slots
            tmp_ptrs[i] = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{TaskRequest}, (Cint,), i-1)
        end
        SLOT_REQUEST_PTRS[] = SVector{MAX_UFI_SLOTS_VAL}(tmp_ptrs)

        LegateInternal._initialize_async_system()

        # Precompile callbacks on main thread
        try
            precompile(ufi_poll_sync, ())
            precompile(_completion_callback, (Int,))
            precompile(_execute_julia_task_internal, (JuliaTaskRequest,))
        catch e
            @warn "Precompilation failed" exception=(e, catch_backtrace())
        end
        UFI_INITIALIZED[] = true
        
        # Start background system
        _start_ufi_threads()
    end
end

const UFI_POLL_INTERVAL = 0.002
const UFI_POLLER_TIMER = Ref{Base.Timer}()

function _start_ufi_threads()
    # Guard against precompilation and multiple starts
    (ccall(:jl_generating_output, Cint, ()) != 0) && return
    UFI_INITIALIZED[] || return
    UFI_SHUTDOWN[] && return
    isassigned(UFI_POLLER_TIMER) && return

    @debug "UFI System: Starting Main-Thread Poller and Workers"
    
    # 1. Main poller on Thread 1
    UFI_POLLER_TIMER[] = Base.Timer(0.0; interval=UFI_POLL_INTERVAL) do timer
        if !UFI_SHUTDOWN[]
            ufi_poll_sync()
        else
            close(timer)
        end
    end

    # 2. Worker threads
    n = Threads.nthreads()
    for i in 1:n
        # In multi-threaded mode, keep Thread 1 free for the poller
        if n > 1 && i == 1
            continue
        end
        # Use standard @spawn for maximum compatibility
        UFI_WORKER_TASKS[i] = errormonitor(Threads.@spawn _ufi_worker_loop())
    end
    UFI_WORKER_COUNT[] = n
end

function shutdown_ufi()
    UFI_SHUTDOWN_DONE[] && return nothing
    UFI_SHUTDOWN[] = true

    if isassigned(JOB_QUEUE) && isopen(JOB_QUEUE[])
        close(JOB_QUEUE[])
    end
    for i in 1:UFI_WORKER_COUNT[]
        if isassigned(UFI_WORKER_TASKS, i)
            try fetch(UFI_WORKER_TASKS[i]) catch end
        end
    end
    UFI_SHUTDOWN_DONE[] = true
end