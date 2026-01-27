using FunctionWrappers
import FunctionWrappers: FunctionWrapper
using Legate

const TaskFunType = FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}

struct JuliaTask
    fun::TaskFunType
end

const my_task_ref = Ref{JuliaTask}()
const my_init_task_ref = Ref{JuliaTask}()

function task_test(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32})
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

function task_init(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32})
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end
end

# Store the wrapped function
const my_task = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}(task_test)
)
const my_init_task = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}(task_init)
)

my_task_ref[] = my_task
my_init_task_ref[] = my_init_task

# Thread-safe execution from Legate worker threads
# Signals via uv_async_send, Julia executes

# Shared data structure for passing task information from C++ to Julia
mutable struct TaskRequest
    task_id::UInt32  # Task ID instead of pointer
    inputs_ptr::Ptr{Ptr{Cvoid}}
    outputs_ptr::Ptr{Ptr{Cvoid}}
    n::Int64

    TaskRequest() = new(0, C_NULL, C_NULL, 0)
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
const ASYNC_COND = Base.AsyncCondition()
const CURRENT_REQUEST = Ref{TaskRequest}(TaskRequest())

# Worker task that waits for async signals from C++
function async_worker()
    @info "AsyncCondition worker started, waiting for UV signals..."
    WORKER_STARTED[] = true

    while true
        try
            # Wait for C++ to call uv_async_send
            @info "Waiting on AsyncCondition..."
            wait(ASYNC_COND)

            @info "✓ Received UV async signal! Executing Julia task..."

            try
                # Get task data (set by C++ before uv_async_send)
                @info "Getting request data"
                req = CURRENT_REQUEST[]
                @info "Request data retrieved" req

                # Execute the Julia task on this Julia thread (SAFE!)
                execute_julia_task(req)

                # Signal completion to C++
                @info "Signaling completion to C++"
                ccall(:completion_callback_from_julia, Cvoid, ())
                @info "Completion signaled"
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

    @info "Step 2: Got task function"

    n = req.n
    @info "Step 3: Allocating arrays" n req.task_id

    # Allocate Julia arrays (safe on this Julia thread!)
    a = Vector{Float32}(undef, n)
    b = Vector{Float32}(undef, n)
    c = Vector{Float32}(undef, n)

    # Task 50001 is init - it only writes outputs, no inputs
    # Task 50002 is compute - it reads inputs and writes output
    if req.task_id == 0xc351  # 50001 in hex
        @info "Init task - loading output pointers only"
        ptr_a = Ptr{Float32}(unsafe_load(req.outputs_ptr, 1))
        ptr_b = Ptr{Float32}(unsafe_load(req.outputs_ptr, 2))
        ptr_c = Ptr{Float32}(unsafe_load(req.outputs_ptr, 3))

        @info "Step 4: Executing init task"
        # Execute task (initializes a, b, c)
        task_fun(a, b, c)

        @info "Step 5: Copying results back (all outputs)"
        # Copy all results back
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            ptr_a, pointer(a), n * sizeof(Float32))
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            ptr_b, pointer(b), n * sizeof(Float32))
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            ptr_c, pointer(c), n * sizeof(Float32))
    else  # 50002 - compute task
        @info "Compute task - loading input and output pointers"
        ptr_a = Ptr{Float32}(unsafe_load(req.inputs_ptr, 1))
        ptr_b = Ptr{Float32}(unsafe_load(req.inputs_ptr, 2))
        ptr_c = Ptr{Float32}(unsafe_load(req.outputs_ptr, 1))

        @info "Step 4: Copying input data"
        # Copy inputs from C++
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            pointer(a), ptr_a, n * sizeof(Float32))
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            pointer(b), ptr_b, n * sizeof(Float32))

        @info "Step 5: Executing compute task"
        # Execute task
        task_fun(a, b, c)

        @info "Step 6: Copying result back"
        # Copy result back
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            ptr_c, pointer(c), n * sizeof(Float32))
    end

    @info "Julia task completed successfully!"

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

# Get the async handle for C++ to call uv_async_send
function get_async_handle()
    return ASYNC_COND.handle
end

# Get pointer to TaskRequest for C++ to write to
function get_request_ptr()
    return Base.unsafe_convert(Ptr{Cvoid}, CURRENT_REQUEST)
end

# Completion callback that C++ will call
Base.@ccallable function completion_callback()::Cvoid
    # This will be called from C++ to signal task completion
    # Just a placeholder for now - C++ uses condition variable
    return nothing
end

# Register completion callback pointer
function register_completion_callback()
    COMPLETION_CALLBACK[] = @cfunction(completion_callback, Cvoid, ())
    return COMPLETION_CALLBACK[]
end

function test_driver()
    rt = Legate.get_runtime()
    lib = Legate.create_library("test")
    Legate.ufi_interface_register(lib)
    println("Registered library with C++ runtime")

    # Initialize the async callback system
    async_handle = get_async_handle()
    request_ptr = get_request_ptr()

    @info "Initializing async system" async_handle request_ptr
    Legate.initialize_async_system(async_handle, request_ptr)
    println("Async system initialized")

    # Wrap Legate operations in @async blocks
    # This keeps the main thread's event loop active!
    @async begin
        task = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

        a = Legate.create_array([100], Float32)
        b = Legate.create_array([100], Float32)
        c = Legate.create_array([100], Float32)

        init_output_vars = Vector{Legate.Variable}()

        push!(init_output_vars, Legate.add_output(task, a))
        push!(init_output_vars, Legate.add_output(task, b))
        push!(init_output_vars, Legate.add_output(task, c))
        Legate.add_scalar(task, Legate.Scalar(UInt32(50001))) # init task id
        task_ptr = Base.unsafe_convert(Ptr{Nothing}, my_init_task_ref)
        Legate.add_scalar(task, Legate.Scalar(task_ptr))
        Legate.default_alignment(task, Vector{Legate.Variable}(), init_output_vars)
        Threads.atomic_add!(PENDING_TASKS, 1)
        Legate.submit_task(rt, task)

        task2 = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

        input_vars = Vector{Legate.Variable}()
        output_vars = Vector{Legate.Variable}()

        push!(input_vars, Legate.add_input(task2, a))
        push!(input_vars, Legate.add_input(task2, b))
        push!(output_vars, Legate.add_output(task2, c))
        Legate.add_scalar(task2, Legate.Scalar(UInt32(50002)))  # compute task id
        task_ptr2 = Base.unsafe_convert(Ptr{Nothing}, my_task_ref)
        Legate.add_scalar(task2, Legate.Scalar(task_ptr2))
        Legate.default_alignment(task2, input_vars, output_vars)
        Threads.atomic_add!(PENDING_TASKS, 1)
        Legate.submit_task(rt, task2)

        println("Task submitted successfully")
    end
end

# Pre-compile the task functions to ensure JIT happens on the main thread
function precompile_tasks()
    println("Pre-compiling tasks...")

    # 1. Standard arrays
    a = zeros(Float32, 1)
    b = zeros(Float32, 1)
    c = zeros(Float32, 1)
    my_init_task_ref[].fun(a, b, c)
    my_task_ref[].fun(a, b, c)

    println("Tasks pre-compiled")
end

if abspath(PROGRAM_FILE) == @__FILE__
    precompile_tasks()

    # Register task functions for thread-safe lookup
    register_task_function(UInt32(50001), my_init_task_ref[].fun)
    register_task_function(UInt32(50002), my_task_ref[].fun)
    @info "Registered task functions: 50001, 50002"

    # Start worker on interactive thread
    # It will wait on Condition variable for signals
    start_worker()

    # Run test driver (submits tasks to Legate)
    test_driver()

    # Keep main thread alive to process events
    # The async worker needs the event loop running to receive signals
    println("Main thread waiting for tasks to complete...")

    lock(ALL_TASKS_DONE) do
        while PENDING_TASKS[] > 0
            # This wait yields to the scheduler, allowing the event loop to run
            # because the scheduler processes UV events when idle/waiting.
            wait(ALL_TASKS_DONE)
        end
    end

    println("Done! All tasks completed.")
end
