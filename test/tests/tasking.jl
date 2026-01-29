function run_base_julia_test()
    a = zeros(Float32, 10, 10)
    b = zeros(Float32, 10, 10)
    c = zeros(Float32, 10, 10)
    d = zeros(Float32, 10, 10)

    # init task equivalent
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end

    a_init = copy(a)
    b_init = copy(b)

    # addition task equivalent (c = a + b)
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end

    # 4-arg task equivalent (inputs: a, c -> outputs: b, d)
    @inbounds @simd for i in eachindex(a)
        b[i] = a[i] * 2
        d[i] = c[i] + 1
    end

    # scalar task equivalent (a = c * 2.5)
    scalar = 2.5f0
    @inbounds @simd for i in eachindex(a)
        a[i] = c[i] * scalar
    end

    return (a=a, b=b, c=c, d=d, a_init=a_init, b_init=b_init)
end

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
    a, b, scalar = args
    # scalar is a scalar (Float32)
    @inbounds @simd for i in eachindex(a)
        b[i] = a[i] * scalar
    end
end

# Helper function to read Legate array data by submitting a copy task
function read_legate_array(rt, lib, legate_arr)
    result = zeros(Float32, 10, 10)

    function copy_task(args::Vector{Legate.TaskArgument})
        src = args[1]
        @inbounds @simd for i in eachindex(src)
            result[i] = src[i]
        end
    end

    copy_task_wrapped = Legate.wrap_task(copy_task)
    task = Legate.create_julia_task(rt, lib, copy_task_wrapped)
    in_vars = Vector{Legate.Variable}()
    push!(in_vars, Legate.add_input(task, legate_arr))
    Legate.default_alignment(task, in_vars, Vector{Legate.Variable}())
    Legate.submit_task(rt, task)
    Legate.wait_ufi()

    return result
end

# get ground truth from base julia
base_results = run_base_julia_test()

# compute expected values from Base Julia results
expected_c = base_results.a_init .+ base_results.b_init
expected_b = base_results.a_init .* 2
expected_d = expected_c .+ 1
expected_a = expected_c .* 2.5f0

@testset "Legate Execution" begin
    rt = Legate.get_runtime()
    lib = Legate.create_library("test_comparison")

    my_task = Legate.wrap_task(task_test)
    my_4arg_task = Legate.wrap_task(task_4arg)
    my_scalar_task = Legate.wrap_task(task_scalar)

    function set_legate_array(rt, lib, legate_arr, values)
        function set_task(args::Vector{Legate.TaskArgument})
            arr = args[1]
            @inbounds @simd for i in eachindex(arr)
                arr[i] = values[i]
            end
        end
        set_wrapped = Legate.wrap_task(set_task)
        task = Legate.create_julia_task(rt, lib, set_wrapped)
        out_vars = Vector{Legate.Variable}()
        push!(out_vars, Legate.add_output(task, legate_arr))
        Legate.default_alignment(task, Vector{Legate.Variable}(), out_vars)
        Legate.submit_task(rt, task)
    end

    @testset "Initialization" begin
        a = Legate.create_array([10, 10], Float32)
        b = Legate.create_array([10, 10], Float32)
        c = Legate.create_array([10, 10], Float32)
        d = Legate.create_array([10, 10], Float32)

        set_legate_array(rt, lib, a, base_results.a_init)
        set_legate_array(rt, lib, b, base_results.b_init)
        set_legate_array(rt, lib, c, zeros(Float32, 10, 10))

        # Verify Init
        val_a = read_legate_array(rt, lib, a)
        val_b = read_legate_array(rt, lib, b)
        @test val_a ≈ base_results.a_init
        @test val_b ≈ base_results.b_init
    end

    a = Legate.create_array([10, 10], Float32)
    b = Legate.create_array([10, 10], Float32)
    c = Legate.create_array([10, 10], Float32)
    d = Legate.create_array([10, 10], Float32)

    set_legate_array(rt, lib, a, base_results.a_init)
    set_legate_array(rt, lib, b, base_results.b_init)
    set_legate_array(rt, lib, c, zeros(Float32, 10, 10))

    @testset "3-Argument Task (c = a + b)" begin
        task2 = Legate.create_julia_task(rt, lib, my_task)
        input_vars = Vector{Legate.Variable}()
        output_vars = Vector{Legate.Variable}()
        push!(input_vars, Legate.add_input(task2, a))
        push!(input_vars, Legate.add_input(task2, b))
        push!(output_vars, Legate.add_output(task2, c))
        Legate.default_alignment(task2, input_vars, output_vars)
        Legate.submit_task(rt, task2)
        val_c = read_legate_array(rt, lib, c)
        @test val_c ≈ expected_c
    end

    @testset "4-Argument Task (Mixing Inputs/Outputs)" begin
        task3 = Legate.create_julia_task(rt, lib, my_4arg_task)
        in_vars_4 = Vector{Legate.Variable}()
        out_vars_4 = Vector{Legate.Variable}()
        push!(in_vars_4, Legate.add_input(task3, a))
        push!(in_vars_4, Legate.add_input(task3, c))
        push!(out_vars_4, Legate.add_output(task3, b))
        push!(out_vars_4, Legate.add_output(task3, d))
        Legate.default_alignment(task3, in_vars_4, out_vars_4)
        Legate.submit_task(rt, task3)
        val_b = read_legate_array(rt, lib, b)
        val_d = read_legate_array(rt, lib, d)
        @test val_b ≈ expected_b
        @test val_d ≈ expected_d
    end

    @testset "Scalar Task (Arg + Scalar)" begin
        task4 = Legate.create_julia_task(rt, lib, my_scalar_task)
        in_vars_s = Vector{Legate.Variable}()
        out_vars_s = Vector{Legate.Variable}()
        push!(in_vars_s, Legate.add_input(task4, c))
        push!(out_vars_s, Legate.add_output(task4, a))
        Legate.add_scalar(task4, Legate.Scalar(2.5f0))
        Legate.default_alignment(task4, in_vars_s, out_vars_s)
        Legate.submit_task(rt, task4)
        val_a = read_legate_array(rt, lib, a)
        @test val_a ≈ expected_a
    end
end
