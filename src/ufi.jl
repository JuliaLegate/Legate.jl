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

    function TaskRequest()
        new(0, 0, 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0, 0, (0, 0, 0))
    end
end

const TASK_REGISTRY = Dict{UInt32,Union{CPUWrapType,Function}}()
const REGISTRY_LOCK = ReentrantLock()
const LAST_CREATED_TASK_ID = Threads.Atomic{Int}(0)

const UFI_INIT_LOCK = ReentrantLock()

const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)
const MAX_SUBMITTED_TASK_ID = Threads.Atomic{Int}(0)

const MAX_UFI_SLOTS = Ref{Int}(0)
const SLOT_WORK_AVAILABLE_PTRS = Vector{Ptr{Int32}}()
const SLOT_REQUEST_PTRS = Vector{Ptr{TaskRequest}}()
const UFI_LAST_TASK_IDS = Vector{Threads.Atomic{UInt64}}()
const UFI_WORKER_TASKS = Vector{Task}()
const JOB_QUEUE = Ref{Channel{Any}}()

const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)
const UFI_EXEC_LOCK = Ref{ReentrantLock}()

const UFI_POLLER_TIMER = Ref{Base.Timer}()
const UFI_POLL_INTERVAL = 0.001 # 1ms

# Serializes first execution of each task_id to avoid JIT race conditions
const JIT_LOCK = ReentrantLock()
const JIT_SEEN = Set{Int}()

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
    Legate.issue_execution_fence(blocking=false)

    while ufi_has_pending_work()
        sleep(0.001) # Yield to background tasks (Timer and Workers)
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

function create_julia_task(rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaCPUTask)
    if task_obj isa JuliaCPUTask
        id = LegateInternal.JULIA_CUSTOM_TASK
        task = LegateInternal.create_auto_task(rt, lib, id)
    else
        id = LegateInternal.JULIA_CUSTOM_GPU_TASK
        task = LegateInternal.create_auto_task(rt, lib, id)
    end
    add_scalar(task, Scalar(Int32(task_obj.task_id)))
    register_task_function(task_obj.task_id, task_obj.fun)
    Threads.atomic_xchg!(LAST_CREATED_TASK_ID, Int(task_obj.task_id))
    return task
end

function _completion_callback(slot_id::Int)
    # Yield to let C++ main thread enter cv.wait before we signal
    yield()
    ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), slot_id)
end

function ufi_poll_sync()
    # Tasks are claimed atomically via CAS, so no global poll guard needed
    try
        if UFI_SHUTDOWN[]
            return false
        end

        tid = Threads.threadid()
        is_main = tid == 1

        if is_main
            if mod(Threads.atomic_add!(UFI_POLL_COUNT, 1), 1000) == 0
                @debug "UFI: Main poller heartbeat (Iteration: $(UFI_POLL_COUNT[]))"
            end
        end

        found_any = false

        # 1. Process JOB_QUEUE (non-blocking)
        while true
            local work_item = nothing
            lock(UFI_EXEC_LOCK[]) do
                if isready(JOB_QUEUE[])
                    work_item = take!(JOB_QUEUE[])
                end
            end

            if isnothing(work_item)
                break # Queue empty
            end

            local slot_id = work_item[1]
            local req = work_item[2]
            local completion_cond = work_item[3]


            try
                _execute_julia_task_internal(req, slot_id)
            catch e
                @error "UFI: Error executing queued task" thread=tid exception=(e, catch_backtrace())
            finally
                if !isnothing(completion_cond)
                    Threads.atomic_set!(completion_cond, true)
                else
                    _completion_callback(slot_id)
                end
            end
            found_any = true
        end

        # 2. Poll Slots
        for i in 1:length(SLOT_WORK_AVAILABLE_PTRS)
            ptr = SLOT_WORK_AVAILABLE_PTRS[i]
            val = UInt64(unsafe_load(ptr))
            Threads.atomic_fence()

            if val != 0
                target_atom = UFI_LAST_TASK_IDS[i]
                last_seen = target_atom[]
                
                # Atomic claim: only one thread proceeds
                if val > last_seen
                    if Threads.atomic_cas!(target_atom, last_seen, val) == last_seen

                        try
                            req = unsafe_load(SLOT_REQUEST_PTRS[i])
                            try
                                _execute_julia_task_internal(req, i-1)
                            catch e
                                @error "UFI: Error executing slotted task" thread=tid slot_id=i-1 work_seq=val exception=(e, catch_backtrace())
                            finally

                                _completion_callback(i-1)
                            end
                            found_any = true
                        catch e
                            @error "UFI: Error processing slotted task" thread=tid slot_id=i-1 work_seq=val exception=(e, catch_backtrace())
                            _completion_callback(i-1)  # Must signal to avoid C++ deadlock
                        end
                    end
                end
            end
        end
        return found_any
    catch e
        @error "UFI: Critical error in ufi_poll_sync" thread=Threads.threadid() exception=(e, catch_backtrace())
        return false
    end
end

function ufi_worker_loop()
    tid = Threads.threadid()
    @debug "UFI Worker $tid started"
    try
        while !UFI_SHUTDOWN[]
            if !ufi_poll_sync()
                sleep(0.001)  # Idle backoff
            end
        end
    catch e
        if !(e isa InvalidStateException)
            @error "UFI: Worker $tid died" exception=(e, catch_backtrace())
        end
    end
end

function _start_ufi_system()
    UFI_SHUTDOWN[] = false
end

function execute_julia_task(@nospecialize(req::TaskRequest), @nospecialize(slot_id::Integer))
    
    # Offload to Julia worker to avoid unadopted-thread issues on C++ threads
    done = Threads.Atomic{Bool}(false)
    lock(UFI_EXEC_LOCK[]) do
        # put! into queue.
        put!(JOB_QUEUE[], (slot_id, req, done))
    end
    while !done[]
        sleep(0.001)
    end
end

function _execute_julia_task_internal(req::TaskRequest, slot_id::Integer)
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = req.task_id == 0 ? ((args...) -> nothing) : TASK_REGISTRY[req.task_id]
    end
    
    if req.is_gpu != 0
        error("Legate UFI: GPU execution not supported.")
    else
        args = Vector{TaskArgument}(undef, Int(req.num_inputs + req.num_outputs + req.num_scalars))
        # Double-checked lock: serialize first execution per task_id for JIT safety
        if req.task_id ∉ JIT_SEEN
            lock(JIT_LOCK) do
                _execute_julia_task_cpu(req, task_fun, args)
                push!(JIT_SEEN, req.task_id)
            end
        else
            _execute_julia_task_cpu(req, task_fun, args)
        end
    end
end

# Field access helpers for TaskRequest
_get_type(x::Ptr{Cint}, i) = unsafe_load(x, i)
_get_ptr(x::Ptr{Ptr{Cvoid}}, i) = unsafe_load(x, i)

function _execute_julia_task_cpu(req::TaskRequest, task_fun::Function, args::Vector{TaskArgument})
    dims = ntuple(i -> Int(max(0, req.dims[i])), req.ndim)
    _fill_args_core!(args, req, dims)
    Base.invokelatest(task_fun, args...)
end

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
        
        JOB_QUEUE[] = Channel{Any}(10000)

        UFI_EXEC_LOCK[] = ReentrantLock()

        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        MAX_UFI_SLOTS[] = max_slots
        
        resize!(UFI_LAST_TASK_IDS, max_slots)
        for i in 1:max_slots
            UFI_LAST_TASK_IDS[i] = Threads.Atomic{UInt64}(0)
        end
        
        empty!(SLOT_WORK_AVAILABLE_PTRS)
        empty!(SLOT_REQUEST_PTRS)
        for i in 1:max_slots
            ptr = ccall((:legate_get_slot_work_available_ptr, Legate.WRAPPER_LIB_PATH), Ptr{Int32}, (Cint,), i-1)
            req_ptr = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{TaskRequest}, (Cint,), i-1)
            push!(SLOT_WORK_AVAILABLE_PTRS, ptr)
            push!(SLOT_REQUEST_PTRS, req_ptr)
        end

        LegateInternal._initialize_async_system()

        # Precompile callbacks on main thread to avoid JIT on worker threads
        try
            precompile(execute_julia_task, (TaskRequest, Int32))
            precompile(execute_julia_task, (TaskRequest, Int64))
            precompile(execute_julia_task, (TaskRequest, Integer))
            precompile(put!, (Channel{Any}, Tuple{Any, Any, Any}))
            precompile(ufi_poll_sync, ())
            precompile(ufi_worker_loop, ())
            precompile(ufi_has_pending_work, ())
            precompile(_completion_callback, (Int,))
        catch e
            @warn "Precompilation failed" exception=(e, catch_backtrace())
        end

        # Force-compile ufi_poll_sync on main thread before any workers touch it.
        # This prevents the JIT race where multiple threads simultaneously
        # compile the same function, crashing in _jl_mutex_wait.
        ufi_poll_sync()

        UFI_INITIALIZED[] = true
    end
    _start_ufi_threads()
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

    if Threads.nthreads() > 1
        env_val = get(ENV, "LEGATE_UFI_WORKERS", nothing)
        n_workers = if isnothing(env_val)
            max(1, Threads.nthreads() - 1)
        else
            parse(Int, env_val)
        end
        
        @debug "UFI System: Spawning $n_workers background hybrid workers"
        for i in 1:n_workers
            push!(UFI_WORKER_TASKS, Threads.@spawn ufi_worker_loop())
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