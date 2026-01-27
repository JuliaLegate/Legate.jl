using Legate

const my_task_ref = Ref{Legate.JuliaTask}()
const my_init_task_ref = Ref{Legate.JuliaTask}()
const my_4arg_task_ref = Ref{Legate.JuliaTask}()
const my_scalar_task_ref = Ref{Legate.JuliaTask}()

function task_test(args::Vector{Legate.TaskArgument})
    a, b, c = args
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

function task_init(args::Vector{Legate.TaskArgument})
    a, b, c = args
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end
end

# Task with 2 inputs, 2 outputs
function task_4arg(args::Vector{Legate.TaskArgument})
    in1, in2, out1, out2 = args
    @inbounds @simd for i in eachindex(in1)
        out1[i] = in1[i] * 2
        out2[i] = in2[i] + 1
    end
end

# Task with Scalar argument
function task_scalar(args::Vector{Legate.TaskArgument})
    a, b, scale = args
    # scale is a scalar (Float32)
    @inbounds @simd for i in eachindex(a)
        b[i] = a[i] * scale
    end
end

# Store the wrapped function
const my_task = Legate.JuliaTask(
    Legate.FunctionWrapper{Nothing,Tuple{Vector{Legate.TaskArgument}}}(task_test)
)
const my_init_task = Legate.JuliaTask(
    Legate.FunctionWrapper{Nothing,Tuple{Vector{Legate.TaskArgument}}}(task_init)
)
const my_4arg_task = Legate.JuliaTask(
    Legate.FunctionWrapper{Nothing,Tuple{Vector{Legate.TaskArgument}}}(task_4arg)
)
const my_scalar_task = Legate.JuliaTask(
    Legate.FunctionWrapper{Nothing,Tuple{Vector{Legate.TaskArgument}}}(task_scalar)
)

my_task_ref[] = my_task
my_init_task_ref[] = my_init_task
my_4arg_task_ref[] = my_4arg_task
my_scalar_task_ref[] = my_scalar_task

function test_driver()
    rt = Legate.get_runtime()
    lib = Legate.create_library("test")

    # Wrap Legate operations in @async blocks
    # This keeps the main thread's event loop active!
    @async begin
        # 1. Init Task (3 args)
        task = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

        a = Legate.create_array([10, 10], Float32)
        b = Legate.create_array([10, 10], Float32)
        c = Legate.create_array([10, 10], Float32)
        d = Legate.create_array([10, 10], Float32) # Extra array for 4-arg test

        init_output_vars = Vector{Legate.Variable}()

        push!(init_output_vars, Legate.add_output(task, a))
        push!(init_output_vars, Legate.add_output(task, b))
        push!(init_output_vars, Legate.add_output(task, c))
        Legate.add_scalar(task, Legate.Scalar(UInt32(50001))) # init task id
        # Task pointer logic is legacy but harmless to keep for now
        task_ptr = Base.unsafe_convert(Ptr{Nothing}, my_init_task_ref)
        Legate.add_scalar(task, Legate.Scalar(task_ptr))
        Legate.default_alignment(task, Vector{Legate.Variable}(), init_output_vars)
        Threads.atomic_add!(Legate.PENDING_TASKS, 1)
        Legate.submit_task(rt, task)

        # 2. Compute Task (3 args)
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

        # 3. Arbitrary Arg Task (4 args: 2 in, 2 out)
        task3 = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

        in_vars_4 = Vector{Legate.Variable}()
        out_vars_4 = Vector{Legate.Variable}()

        # Inputs: a, c
        push!(in_vars_4, Legate.add_input(task3, a))
        push!(in_vars_4, Legate.add_input(task3, c))
        # Outputs: b (reuse), d (new)
        push!(out_vars_4, Legate.add_output(task3, b))
        push!(out_vars_4, Legate.add_output(task3, d))

        Legate.add_scalar(task3, Legate.Scalar(UInt32(50003))) # 4-arg task id
        task_ptr3 = Base.unsafe_convert(Ptr{Nothing}, my_4arg_task_ref)
        Legate.add_scalar(task3, Legate.Scalar(task_ptr3))
        Legate.default_alignment(task3, in_vars_4, out_vars_4)
        Threads.atomic_add!(Legate.PENDING_TASKS, 1)
        Legate.submit_task(rt, task3)

        # 4. Scalar Arg Task (2 args + scalar)
        task4 = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

        in_vars_s = Vector{Legate.Variable}()
        out_vars_s = Vector{Legate.Variable}()

        # Input: c (result of task2)
        push!(in_vars_s, Legate.add_input(task4, c))
        # Output: a (reuse)
        push!(out_vars_s, Legate.add_output(task4, a))

        Legate.add_scalar(task4, Legate.Scalar(UInt32(50004))) # scalar task id
        task_ptr4 = Base.unsafe_convert(Ptr{Nothing}, my_scalar_task_ref)
        Legate.add_scalar(task4, Legate.Scalar(task_ptr4))

        # Add user scalar argument (Float32)
        # Assuming Legate.Scalar can wrap generic types
        Legate.add_scalar(task4, Legate.Scalar(2.5f0))

        Legate.default_alignment(task4, in_vars_s, out_vars_s)
        Threads.atomic_add!(Legate.PENDING_TASKS, 1)
        Legate.submit_task(rt, task4)

        println("Tasks submitted successfully")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Register task functions for thread-safe lookup
    Legate.register_task_function(UInt32(50001), my_init_task_ref[].fun)
    Legate.register_task_function(UInt32(50002), my_task_ref[].fun)
    Legate.register_task_function(UInt32(50003), my_4arg_task_ref[].fun)
    Legate.register_task_function(UInt32(50004), my_scalar_task_ref[].fun)
    @info "Registered task functions: 50001, 50002, 50003, 50004"

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
