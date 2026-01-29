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
    task_id::UInt32
    input_types::Vector{DataType}
    output_types::Vector{DataType}
end

function wrap_task(f; input_types=DataType[], output_types=DataType[])
    JuliaTask(
        FunctionWrapper{Nothing,Tuple{Vector{TaskArgument}}}(f),
        Threads.atomic_add!(NEXT_TASK_ID, UInt32(1)),
        input_types,
        output_types,
    )
end

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

    TaskRequest() = new(0, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0, 0, (0, 0, 0))
end

# Thread-safe task registry
const TASK_REGISTRY = Dict{UInt32,TaskFunType}()
const REGISTRY_LOCK = ReentrantLock()
const TASK_ARG_TYPES = Dict{UInt32,Tuple{Vector{DataType},Vector{DataType}}}()

# Atomic counter for auto-generating task IDs
const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)

# Task Synchronization
const PENDING_TASKS = Threads.Atomic{Int}(0)
const ALL_TASKS_DONE = Threads.Condition()

# Track if UFI has been shut down
const UFI_SHUTDOWN_DONE = Threads.Atomic{Bool}(false)

function register_task_function(id::UInt32, fun::TaskFunType)
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[id] = fun
    end
    Threads.atomic_add!(Legate.PENDING_TASKS, 1)
end

"""
    create_julia_task(rt, lib, task_obj::Ref{JuliaTask}) -> AutoTask

Create a Julia UFI task with auto-generated task ID.
Automatically registers the task with Legate.
"""
function create_julia_task(
    rt, lib, task_obj::JuliaTask; input_types=DataType[], output_types=DataType[]
)
    # Create the task
    task = create_task(rt, lib, JULIA_CUSTOM_TASK)

    # Add task ID as first scalar
    add_scalar(task, Scalar(task_obj.task_id))

    # Register the function with TASK_REGISTRY
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[task_obj.task_id] = task_obj.fun

        # Prefer kwargs if provided (override), otherwise use task object's types
        in_types = isempty(input_types) ? task_obj.input_types : input_types
        out_types = isempty(output_types) ? task_obj.output_types : output_types

        TASK_ARG_TYPES[task_obj.task_id] = (in_types, out_types)
    end
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
    WORKER_STARTED[] = true
    try
        while true
            wait(ASYNC_COND[])
            req = CURRENT_REQUEST[]
            execute_julia_task(req)
            ccall(:completion_callback_from_julia, Cvoid, ())
        end
    catch e
        isa(e, EOFError) || rethrow()
    end
end

# Execute the actual Julia task logic
function execute_julia_task(req::TaskRequest)
    # Look up task function by ID (thread-safe)
    local task_fun
    local input_types::Vector{DataType}
    local output_types::Vector{DataType}

    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[req.task_id]
        (input_types, output_types) = get(TASK_ARG_TYPES, req.task_id, (DataType[], DataType[]))
    end

    try
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
            T = length(input_types) >= i ? input_types[i] : Float32 # default to Float32 if no type
            ptr = Ptr{T}(unsafe_load(req.inputs_ptr, i))
            push!(args, unsafe_wrap(Array, ptr, dims))
        end

        for i in 1:req.num_outputs
            T = length(output_types) >= i ? output_types[i] : Float32 # default to Float32 if no type
            ptr = Ptr{T}(unsafe_load(req.outputs_ptr, i))
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

        @debug "Legate UFI: Julia task completed successfully!" req.task_id

    catch e
        @error "Legate UFI: Julia task failed" exception=(e, catch_backtrace()) req.task_id
        rethrow()
    finally
        val = Threads.atomic_sub!(PENDING_TASKS, 1)
        if val == 1 # atomic_sub returns OLD value, so if old was 1, new is 0
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
    # Spawn worker on interactive thread pool
    WORKER_TASK[] = Threads.@spawn :interactive async_worker()

    @debug "Legate UFI: Worker task spawned on interactive thread"

    # Wait until worker is ready
    while !WORKER_STARTED[]
        sleep(0.01)
    end

    @debug "Legate UFI: Worker confirmed started and waiting"
end

# Initialize AsyncCondition and start worker - called from Legate.__init__
function init_ufi()
    ASYNC_COND[] = Base.AsyncCondition()
    CURRENT_REQUEST[] = TaskRequest()
    _start_worker()
    sleep(0.1)
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

function shutdown_ufi()
    # Prevent double shutdown
    UFI_SHUTDOWN_DONE[] && return nothing
    UFI_SHUTDOWN_DONE[] = true
    close(ASYNC_COND[])
    wait(WORKER_TASK[])
end
