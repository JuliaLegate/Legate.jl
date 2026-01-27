using Legate

const my_task_ref = Ref{Legate.JuliaTask}()
const my_init_task_ref = Ref{Legate.JuliaTask}()

function task_test(a::AbstractArray, b::AbstractArray, c::AbstractArray)
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

function task_init(a::AbstractArray, b::AbstractArray, c::AbstractArray)
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end
end

# Store the wrapped function
const my_task = Legate.JuliaTask(
    Legate.FunctionWrapper{Nothing,Tuple{AbstractArray,AbstractArray,AbstractArray}}(task_test)
)
const my_init_task = Legate.JuliaTask(
    Legate.FunctionWrapper{Nothing,Tuple{AbstractArray,AbstractArray,AbstractArray}}(task_init)
)

my_task_ref[] = my_task
my_init_task_ref[] = my_init_task

function test_driver()
    rt = Legate.get_runtime()
    lib = Legate.create_library("test")
    Legate.ufi_interface_register(lib)
    println("Registered library with C++ runtime")

    # Initialize the async callback system
    # Use Legate accessors
    async_handle = Legate.get_async_handle()
    request_ptr = Legate.get_request_ptr()

    @info "Initializing async system" async_handle request_ptr
    Legate.initialize_async_system(async_handle, request_ptr)
    println("Async system initialized")

    # Wrap Legate operations in @async blocks
    # This keeps the main thread's event loop active!
    @async begin
        task = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

        a = Legate.create_array([10, 10], Float32)
        b = Legate.create_array([10, 10], Float32)
        c = Legate.create_array([10, 10], Float32)

        init_output_vars = Vector{Legate.Variable}()

        push!(init_output_vars, Legate.add_output(task, a))
        push!(init_output_vars, Legate.add_output(task, b))
        push!(init_output_vars, Legate.add_output(task, c))
        Legate.add_scalar(task, Legate.Scalar(UInt32(50001))) # init task id
        task_ptr = Base.unsafe_convert(Ptr{Nothing}, my_init_task_ref)
        Legate.add_scalar(task, Legate.Scalar(task_ptr))
        Legate.default_alignment(task, Vector{Legate.Variable}(), init_output_vars)
        Threads.atomic_add!(Legate.PENDING_TASKS, 1)
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
        Threads.atomic_add!(Legate.PENDING_TASKS, 1)
        Legate.submit_task(rt, task2)

        println("Task submitted successfully")
    end
end

# Pre-compile the task functions to ensure JIT happens on the main thread
function precompile_tasks()
    println("Pre-compiling tasks...")

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
    Legate.register_task_function(UInt32(50001), my_init_task_ref[].fun)
    Legate.register_task_function(UInt32(50002), my_task_ref[].fun)
    @info "Registered task functions: 50001, 50002"

    # Worker already started by Legate.__init__
    test_driver()

    println("Main thread waiting for tasks to complete...")

    lock(Legate.ALL_TASKS_DONE) do
        while Legate.PENDING_TASKS[] > 0
            wait(Legate.ALL_TASKS_DONE)
        end
    end

    println("Done! All tasks completed.")
end
