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
    is_gpu::Cint # Use Cint for better alignment
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

# Thread-safe task registry
# Union{CPUWrapType,Function} to allow storing both CPU FunctionWrappers and GPU kernel functions
const TASK_REGISTRY = Dict{UInt32,Union{CPUWrapType,Function}}()
const REGISTRY_LOCK = ReentrantLock()

# Atomic counter for auto-generating task IDs
const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)

# Task Synchronization
const PENDING_TASKS = Threads.Atomic{Int}(0)
const ALL_TASKS_DONE = Threads.Condition()

# Track if UFI has been shut down
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)

function register_task_function(id::UInt32, fun::Union{CPUWrapType,Function})
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[id] = fun
    end
    Threads.atomic_add!(Legate.PENDING_TASKS, 1)
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

# Global state
# We use a Ref{TaskRequest} to provide stable memory for C++
# Ref{T} for bits types (immutable structs) holds the data inline.
const CURRENT_REQUEST = Ref{TaskRequest}()

# Worker task that waits for async signals from C++
function async_worker()
    WORKER_STARTED[] = true
    @debug "Legate UFI: Worker started on thread $(Threads.threadid())"
    try
        while !UFI_SHUTDOWN_DONE[]
            ready = ccall(:legate_poll_work, Cint, ()) != 0
            if ready
                req = CURRENT_REQUEST[]
                try
                    execute_julia_task(req)
                catch e
                    @error "Ufi Worker: task failed" exception=(e, catch_backtrace())
                finally
                    ccall(:completion_callback_from_julia, Cvoid, ())
                end
            else
                yield()
            end
        end
    catch e
        isa(e, EOFError) || rethrow()
    end
end

# in CUDAExt ufi.jl
# function _execute_julia_task(::Val{:gpu}, req, task_fun) end
function _execute_julia_task(::Val{:cpu}, req, task_fun)
    args = Vector{TaskArgument}()
    sizehint!(args, req.num_inputs + req.num_outputs + req.num_scalars)

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

    cpu_args = Vector{TaskArgument}(args)
    task_fun(cpu_args)
end

bool_to_symbol(is_gpu::Bool) = is_gpu ? :gpu : :cpu

function execute_julia_task(req::TaskRequest)
    # Look up task function by ID (thread-safe)
    local task_fun

    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[req.task_id]
    end

    try
        Base.invokelatest(_execute_julia_task, Val(bool_to_symbol(req.is_gpu != 0)), req, task_fun)
        yield()
    catch e
        @error "Legate UFI: Julia task failed" exception=(e, catch_backtrace()) req.task_id
        rethrow()
    finally
        # task is done, decrement counter
        val = Threads.atomic_sub!(PENDING_TASKS, 1)
        if val[] == 1 # atomic_sub returns OLD value, so if old was 1, new is 0
            lock(ALL_TASKS_DONE) do
                notify(ALL_TASKS_DONE)
            end
        end
    end
end

# Start the worker
const WORKER_TASK = Ref{Task}()
const WORKER_STARTED = Ref{Bool}(false)

function _start_worker()
    # Spawn worker - will run on any available thread, or interleave on main thread
    WORKER_TASK[] = errormonitor(Threads.@spawn async_worker())

    @debug "Legate UFI: Worker task spawned"

    # Wait until worker is ready
    while !WORKER_STARTED[]
        sleep(0.01)
    end

    @debug "Legate UFI: Worker confirmed started and waiting"
end

# Initialize and start worker on INTERACTIVE thread loop
function init_ufi()
    request_ptr = Legate._get_request_ptr()
    Legate._initialize_async_system(request_ptr)

    init_task = Threads.@spawn :interactive begin
        CURRENT_REQUEST[] = TaskRequest()
        _start_worker()
    end
    wait(init_task)
end

function wait_ufi()
    # For single-threaded Julia, we need to manually poll and execute tasks
    # because blocking in wait() prevents async_worker from running
    if Threads.nthreads() == 1
        # Manual polling loop for single-threaded execution
        while Legate.PENDING_TASKS[] > 0
            # Sleep to allow async_worker to wake up and process C++ signals
            sleep(0.001)  # 1ms - lets async_worker run
            yield()

            # Check for work and execute it (non-blocking poll)
            if ccall(:legate_poll_work, Cint, ()) != 0
                try
                    req = Legate.CURRENT_REQUEST[]
                    Legate.execute_julia_task(req)
                    ccall(:completion_callback_from_julia, Cvoid, ())
                catch e
                    @error "UFI task execution failed" exception=(e, catch_backtrace())
                    rethrow()
                end
            end
        end
    else
        # Multi-threaded: use condition variable (async_worker runs on separate thread)
        lock(Legate.ALL_TASKS_DONE) do
            while Legate.PENDING_TASKS[] > 0
                wait(Legate.ALL_TASKS_DONE)
            end
        end
    end
end

# Get pointer to TaskRequest for C++ to write to
function _get_request_ptr()
    return Base.unsafe_convert(Ptr{Cvoid}, CURRENT_REQUEST)
end

function shutdown_ufi()
    # Prevent double shutdown
    UFI_SHUTDOWN_DONE[] && return nothing
    UFI_SHUTDOWN_DONE[] = true
    wait(WORKER_TASK[])
end
