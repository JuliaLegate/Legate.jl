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

using FunctionWrappers
import FunctionWrappers: FunctionWrapper

# Legate scalar types + AbstractArray for tasks
const TaskArgument = Union{AbstractArray,SUPPORTED_TYPES}

const TaskFunType = FunctionWrapper{Nothing,Tuple{Vector{TaskArgument}}}

struct JuliaTask
    fun::TaskFunType
end

wrap_task(f) = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{TaskArgument}}}(f)
)

# Thread-safe execution from Legate worker threads
# Signals via uv_async_send, Julia executes

# Shared data structure for passing task information from C++ to Julia
struct TaskRequest
    task_id::UInt32
    inputs_ptr::Ptr{Ptr{Cvoid}}
    outputs_ptr::Ptr{Ptr{Cvoid}}
    scalars_ptr::Ptr{Ptr{Cvoid}}
    scalar_types::Ptr{Cint}
    num_inputs::Csize_t
    num_outputs::Csize_t
    num_scalars::Csize_t
    ndim::Cint
    dims::NTuple{3,Int64}

    # Default constructor for initialization
    TaskRequest() = new(0, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0, 0, (0, 0, 0))
end

# Thread-safe task registry
const TASK_REGISTRY = Dict{UInt32,TaskFunType}()
const REGISTRY_LOCK = ReentrantLock()

# Atomic counter for auto-generating task IDs
const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)

# Task Synchronization
const PENDING_TASKS = Threads.Atomic{Int}(0)
const ALL_TASKS_DONE = Threads.Condition()

function register_task_function(id::UInt32, fun::TaskFunType)
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[id] = fun
    end
end

"""
    create_julia_task(rt, lib, task_fun::TaskFunType) -> AutoTask

Create a Julia UFI task with auto-generated task ID.
Automatically registers the task function and adds the task ID as a scalar.
"""
function create_julia_task(rt, lib, task_fun::TaskFunType)
    # Generate unique task ID
    task_id = Threads.atomic_add!(NEXT_TASK_ID, UInt32(1))

    # Create the task
    task = create_task(rt, lib, JULIA_CUSTOM_TASK)

    # Add task ID as first scalar
    add_scalar(task, Scalar(task_id))

    # Register the function
    register_task_function(task_id, task_fun)

    # Increment pending tasks
    Threads.atomic_add!(Legate.PENDING_TASKS, 1)

    return task
end

# Global state
const ASYNC_COND = Ref{Base.AsyncCondition}()
# We use a Ref{TaskRequest} to provide stable memory for C++
# Ref{T} for bits types (immutable structs) holds the data inline.
const CURRENT_REQUEST = Ref{TaskRequest}()

# Worker task that waits for async signals from C++
function async_worker()
    @info "AsyncCondition worker started, waiting for UV signals..."
    WORKER_STARTED[] = true

    while true
        try
            # Wait for C++ to call uv_async_send
            @info "Waiting on AsyncCondition..."
            wait(ASYNC_COND[])

            @info "Received UV async signal! Executing Julia task..."
            try
                # Get task data (set by C++ before uv_async_send)
                req = CURRENT_REQUEST[]

                execute_julia_task(req)
                ccall(:completion_callback_from_julia, Cvoid, ())
            catch e
                @error "CRASH in worker" exception=(e, catch_backtrace())
                rethrow()
            end
        catch e
            @error "Error in worker" exception=(e, catch_backtrace())
        end
    end
end

# Execute the actual Julia task logic
function execute_julia_task(req::TaskRequest)
    # Look up task function by ID (thread-safe)
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[req.task_id]
    end

    # Dynamic argument collection
    args = TaskArgument[]

    # Construct shape tuple
    dims = if req.ndim == 1
        (req.dims[1],)
    elseif req.ndim == 2
        (req.dims[1], req.dims[2])
    elseif req.ndim == 3
        (req.dims[1], req.dims[2], req.dims[3])
    else
        # fallback to 1D if dims exist
        (req.dims[1],)
    end

    for i in 1:req.num_inputs
        ptr = Ptr{Float32}(unsafe_load(req.inputs_ptr, i))
        push!(args, unsafe_wrap(Array, ptr, dims))
    end

    for i in 1:req.num_outputs
        ptr = Ptr{Float32}(unsafe_load(req.outputs_ptr, i))
        push!(args, unsafe_wrap(Array, ptr, dims))
    end

    for i in 1:req.num_scalars
        val_ptr = unsafe_load(req.scalars_ptr, i)
        type_code = Int(unsafe_load(req.scalar_types, i))

        val = if haskey(code_type_map, type_code)
            T = code_type_map[type_code]
            unsafe_load(Ptr{T}(val_ptr))
        else
            @warn "Unknown scalar type code" type_code
            nothing
        end
        push!(args, val)
    end

    task_fun(args)

    @debug "Julia task completed successfully!" req.task_id

    val = Threads.atomic_sub!(PENDING_TASKS, 1)
    if val == 1 # atomic_sub returns OLD value, so if old was 1, new is 0
        lock(ALL_TASKS_DONE) do
            notify(ALL_TASKS_DONE)
        end
    end
end

# Start the worker
const WORKER_TASK = Ref{Task}()
const WORKER_STARTED = Ref{Bool}(false)

function _start_worker()
    # Spawn worker on interactive thread pool
    WORKER_TASK[] = Threads.@spawn :interactive async_worker()

    @info "Worker task spawned on interactive thread"

    # Wait until worker is ready
    while !WORKER_STARTED[]
        sleep(0.01)
    end

    @info "Worker confirmed started and waiting"
end

# Initialize AsyncCondition and start worker - called from Legate.__init__
function init_ufi()
    ASYNC_COND[] = Base.AsyncCondition()
    CURRENT_REQUEST[] = TaskRequest()
    _start_worker()
end

# Wait for all tasks to complete
function wait_ufi()
    lock(Legate.ALL_TASKS_DONE) do
        while Legate.PENDING_TASKS[] > 0
            wait(Legate.ALL_TASKS_DONE)
        end
    end
end

# Get the async handle for C++ to call uv_async_send
function _get_async_handle()
    return ASYNC_COND[].handle
end

# Get pointer to TaskRequest for C++ to write to
function _get_request_ptr()
    return Base.unsafe_convert(Ptr{Cvoid}, CURRENT_REQUEST)
end
