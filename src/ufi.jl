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

# allignment contrainsts are transitive.
# we can allign all the inputs and then alligns all the outputs
# then allign one input with one output
# This reduces the need for a cartesian product.
function default_alignment(
    task::Legate.AutoTask, inputs::Vector{Legate.Variable}, outputs::Vector{Legate.Variable}
)
    # Align all inputs to the first input
    for i in 2:length(inputs)
        Legate.add_constraint(task, Legate.align(inputs[i], inputs[1]))
    end
    # Align all outputs to the first output
    for i in 2:length(outputs)
        Legate.add_constraint(task, Legate.align(outputs[i], outputs[1]))
    end
    # Align first output with first input
    if !isempty(inputs) && !isempty(outputs)
        Legate.add_constraint(task, Legate.align(outputs[1], inputs[1]))
    end
end

const TaskFunType = FunctionWrapper{Nothing,Tuple{AbstractArray,AbstractArray,AbstractArray}}

struct JuliaTask
    fun::TaskFunType
end

# Thread-safe execution from Legate worker threads
# Signals via uv_async_send, Julia executes

# Shared data structure for passing task information from C++ to Julia
mutable struct TaskRequest
    task_id::UInt32  # Task ID instead of pointer
    inputs_ptr::Ptr{Ptr{Cvoid}}
    outputs_ptr::Ptr{Ptr{Cvoid}}
    num_inputs::Csize_t
    num_outputs::Csize_t
    ndim::Cint
    dims::NTuple{3,Int64}

    TaskRequest() = new(0, C_NULL, C_NULL, 0, 0, 0, (0, 0, 0))
end

# Thread-safe task registry
const TASK_REGISTRY = Dict{UInt32,TaskFunType}()
const REGISTRY_LOCK = ReentrantLock()

# Task Synchronization
const PENDING_TASKS = Threads.Atomic{Int}(0)
const ALL_TASKS_DONE = Threads.Condition()

function register_task_function(id::UInt32, fun::TaskFunType)
    lock(REGISTRY_LOCK) do
        TASK_REGISTRY[id] = fun
    end
end

# Global state
const ASYNC_COND = Ref{Base.AsyncCondition}()
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
    @info "Step 1: Looking up task by ID" req.task_id

    # Look up task function by ID (thread-safe)
    local task_fun
    lock(REGISTRY_LOCK) do
        task_fun = TASK_REGISTRY[req.task_id]
    end

    @info "Step 2: Got task function" req.task_id

    # Dynamic argument collection
    args = Vector{AbstractArray}()

    # Construct shape tuple
    dims = if req.ndim == 1
        (req.dims[1],)
    elseif req.ndim == 2
        (req.dims[1], req.dims[2])
    elseif req.ndim == 3
        (req.dims[1], req.dims[2], req.dims[3])
    else
        @error("Unknown number of dimensions: $(req.ndim)")
    end

    @info "Step 3: Constructing shape tuple" dims

    # 1. Process inputs
    for i in 1:req.num_inputs
        ptr = Ptr{Float32}(unsafe_load(req.inputs_ptr, i))
        push!(args, unsafe_wrap(Array, ptr, dims))
    end

    # 2. Process outputs
    for i in 1:req.num_outputs
        ptr = Ptr{Float32}(unsafe_load(req.outputs_ptr, i))
        push!(args, unsafe_wrap(Array, ptr, dims))
    end

    @info "Step 3: Executing task with $(length(args)) arguments of shape $dims"

    # Execute task with splatted arguments
    task_fun(args...)

    @info "Julia task completed successfully!" req.task_id

    # Decrement pending task counter and notify if empty
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

function start_worker()
    # Spawn worker on interactive thread pool
    WORKER_TASK[] = Threads.@spawn :interactive async_worker()

    @info "Worker task spawned on interactive thread"

    # Wait until worker is ready
    while !WORKER_STARTED[]
        sleep(0.01)
    end

    @info "Worker confirmed started and waiting"
end

# Initialize AsyncCondition and start worker - called from __init__
function init_ufi()
    ASYNC_COND[] = Base.AsyncCondition()
    CURRENT_REQUEST[] = TaskRequest() # Runtime allocation
    start_worker()
end

# Get the async handle for C++ to call uv_async_send
function get_async_handle()
    return ASYNC_COND[].handle
end

# Get pointer to TaskRequest for C++ to write to
function get_request_ptr()
    return Base.unsafe_convert(Ptr{Cvoid}, CURRENT_REQUEST)
end
