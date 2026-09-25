# Operands must be large enough that Legate would otherwise split them.

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

@testset verbose=true "Tasking Constraints" begin
    Legate.Experimental(true)  # tasking is experimental
    my_stencil_task = Legate.wrap_task(stencil_task, Legate.CPUBackend)
    my_gather_task = Legate.wrap_task(gather_task, Legate.CPUBackend)
    my_row_normalize_task = Legate.wrap_task(row_normalize_task, Legate.CPUBackend)

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
