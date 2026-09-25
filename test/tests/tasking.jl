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

function task_test(a, b, c)
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

# Task with 2 inputs, 2 outputs
# Task with 2 inputs, 2 outputs
function task_4arg(in1, in2, out1, out2)
    @inbounds @simd for i in eachindex(in1)
        out1[i] = in1[i] * 2
        out2[i] = in2[i] + 1
    end
end

# Task with Scalar argument
# Task with Scalar argument
function task_scalar(a, b, scalar)
    # scalar is a scalar (Float32)
    @inbounds @simd for i in eachindex(a)
        b[i] = a[i] * scalar
    end
end

function stencil_task(core_indices, halo, output)
    offset = Int(core_indices[1] - halo[1])
    @assert length(halo) >= length(output)
    @assert 0 <= offset <= 1
    @inbounds for i in eachindex(output)
        h = i + offset
        center = halo[h]
        left = h == 1 ? center : halo[h - 1]
        right = h == length(halo) ? center : halo[h + 1]
        output[i] = left + 2 * center + right
    end
    return nothing
end

function gather_task(indices, table, out)
    @inbounds for i in eachindex(out)
        out[i] = table[Int(indices[i])]
    end
    return nothing
end

function row_normalize_task(a, out)
    @inbounds for i in axes(a, 1)
        s = zero(eltype(a))
        for j in axes(a, 2)
            s += a[i, j]
        end
        for j in axes(a, 2)
            out[i, j] = a[i, j] / s
        end
    end
    return nothing
end

# get ground truth from base julia
base_results = run_base_julia_test()

# compute expected values from Base Julia results
expected_c = base_results.a_init .+ base_results.b_init
expected_b = base_results.a_init .* 2
expected_d = expected_c .+ 1
expected_a = expected_c .* 2.5f0

@testset verbose=true "CPU Tasking" begin
    Legate.Experimental(true)  # tasking is experimental
    rt = Legate.get_runtime()
    lib = Legate.create_library("test_comparison")

    my_task = Legate.wrap_task(task_test, Legate.CPUBackend)
    my_4arg_task = Legate.wrap_task(task_4arg, Legate.CPUBackend)
    my_scalar_task = Legate.wrap_task(task_scalar, Legate.CPUBackend)
    my_stencil_task = Legate.wrap_task(stencil_task, Legate.CPUBackend)
    my_gather_task = Legate.wrap_task(gather_task, Legate.CPUBackend)
    my_row_normalize_task = Legate.wrap_task(row_normalize_task, Legate.CPUBackend)

    @testset "Initialization" begin
        a = Legate.create_array([10, 10], Float32)
        b = Legate.create_array([10, 10], Float32)
        c = Legate.create_array([10, 10], Float32)
        d = Legate.create_array([10, 10], Float32)

        # Use copyto! instead of set_task closure
        copyto!(a, base_results.a_init)
        copyto!(b, base_results.b_init)
        copyto!(c, zeros(Float32, 10, 10))

        # Verify Init
        val_a = Array(a)
        val_b = Array(b)
        @test val_a ≈ base_results.a_init
        @test val_b ≈ base_results.b_init
    end

    @testset "LogicalArray Slice" begin
        reference = reshape(Float32.(1:36), 6, 6)
        array = Legate.create_array([6, 6], Float32)
        copyto!(array, reference)
        view = Legate.slice(array, 0, 1, 5)
        view = Legate.slice(view, 1, 2, 6)
        @test size(view) == (4, 4)
    end

    a = Legate.create_array([10, 10], Float32)
    b = Legate.create_array([10, 10], Float32)
    c = Legate.create_array([10, 10], Float32)
    d = Legate.create_array([10, 10], Float32)
    copyto!(a, base_results.a_init)
    copyto!(b, base_results.b_init)
    copyto!(c, zeros(Float32, 10, 10))

    @testset "3-Argument Task (c = a + b)" begin
        task2 = Legate.create_julia_task(rt, lib, my_task)
        input_vars = Vector{Legate.Variable}()
        output_vars = Vector{Legate.Variable}()
        push!(input_vars, Legate.add_input(task2, a))
        push!(input_vars, Legate.add_input(task2, b))
        push!(output_vars, Legate.add_output(task2, c))
        Legate.default_alignment(task2, input_vars, output_vars)
        Legate.submit_task(rt, task2)
        val_c = Array(c)
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
        val_b = Array(b)
        val_d = Array(d)
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
        val_a = Array(a)
        @test val_a ≈ expected_a
    end

    @testset "Execution Fence" begin
        @test Legate.issue_execution_fence() === nothing
        @test Legate.runtime_sync() === nothing
    end

    @testset "Bloat Constraint (Radius-One Stencil)" begin
        if !isdefined(Legate.LegateInternal, :bloat)
            @test_skip false
        else
            n = 100_000
            reference = Float64.(1:n)
            core_indices = Legate.create_array([n], Float64)
            stencil_input = Legate.create_array([n], Float64)
            stencil_output = Legate.create_array([n], Float64)
            copyto!(core_indices, reference)
            copyto!(stencil_input, reference)

            task = Legate.create_julia_task(rt, lib, my_stencil_task)
            core_var = Legate.add_input(task, core_indices)
            halo_var = Legate.add_input(task, stencil_input)
            output_var = Legate.add_output(task, stencil_output)
            Legate.add_constraint(task, Legate.align(core_var, output_var))
            Legate.add_constraint(task, Legate.bloat(core_var, halo_var, (1,), (1,)))

            started_before = Legate.LegateInternal.legate_get_started_count()
            Legate.submit_task(rt, task)
            result = Array(stencil_output)
            started =
                Legate.LegateInternal.legate_get_started_count() - started_before

            expected = 4 .* reference
            expected[1] = 5
            expected[end] = 4n - 1
            @test result == expected
            if Legate.LegateInternal.num_procs() > 1
                @test started > 1
            end
        end
    end

    # Operands must be large enough that Legate would otherwise split them.
    @testset "Broadcast Constraint (Full)" begin
        n = 100_000
        indices_host = Float64.(n:-1:1)
        table_host = Float64.(1:n) .^ 2
        indices = Legate.create_array([n], Float64)
        table = Legate.create_array([n], Float64)
        out = Legate.create_array([n], Float64)
        copyto!(indices, indices_host)
        copyto!(table, table_host)

        task = Legate.create_julia_task(rt, lib, my_gather_task)
        indices_var = Legate.add_input(task, indices)
        table_var = Legate.add_input(task, table)
        out_var = Legate.add_output(task, out)
        Legate.add_constraint(task, Legate.align(indices_var, out_var))
        Legate.add_constraint(task, Legate.broadcast(table_var))

        started_before = Legate.LegateInternal.legate_get_started_count()
        Legate.submit_task(rt, task)
        result = Array(out)
        started = Legate.LegateInternal.legate_get_started_count() - started_before

        @test result == reverse(table_host)
        if Legate.LegateInternal.num_procs() > 1
            @test started > 1
        end
    end

    @testset "Broadcast Constraint (Axes)" begin
        m, k = 1024, 1024
        a_host = reshape(Float64.(1:(m * k)), m, k)
        a = Legate.create_array([m, k], Float64)
        out = Legate.create_array([m, k], Float64)
        copyto!(a, a_host)

        task = Legate.create_julia_task(rt, lib, my_row_normalize_task)
        a_var = Legate.add_input(task, a)
        out_var = Legate.add_output(task, out)
        Legate.add_constraint(task, Legate.align(a_var, out_var))
        # Row sums need every column in each tile.
        Legate.add_constraint(task, Legate.broadcast(a_var, (1,)))

        started_before = Legate.LegateInternal.legate_get_started_count()
        Legate.submit_task(rt, task)
        result = Array(out)
        started = Legate.LegateInternal.legate_get_started_count() - started_before

        @test result ≈ a_host ./ sum(a_host; dims=2)
        if Legate.LegateInternal.num_procs() > 1
            @test started > 1
        end
    end
end
