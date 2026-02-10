using Legate

function task_test(a, b, c)
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
    @info "task_test executed"
end

function task_init(a, b, c)
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end
    @info "task_init executed"
end

function task_4arg(in1, in2, out1, out2)
    @inbounds @simd for i in eachindex(in1)
        out1[i] = in1[i] * 2
        out2[i] = in2[i] + 1
    end
    @info "task_4arg executed"
end

function task_scalar(a, b, scalar)
    @inbounds @simd for i in eachindex(a)
        b[i] = a[i] * scalar
    end
    @info "task_scalar executed"
end

function test_driver()
    N = 1000
    rt = Legate.get_runtime()
    lib = Legate.create_library("test")

    my_task = Legate.wrap_task(task_test)
    my_init_task = Legate.wrap_task(task_init)
    my_4arg_task = Legate.wrap_task(task_4arg)
    my_scalar_task = Legate.wrap_task(task_scalar)

    # 1. Init Task (3 args)
    a = Legate.create_array([N, N], Float32)
    b = Legate.create_array([N, N], Float32)
    c = Legate.create_array([N, N], Float32)
    d = Legate.create_array([N, N], Float32)

    task = Legate.create_julia_task(rt, lib, my_init_task)
    init_output_vars = Vector{Legate.Variable}()
    push!(init_output_vars, Legate.add_output(task, a))
    push!(init_output_vars, Legate.add_output(task, b))
    push!(init_output_vars, Legate.add_output(task, c))
    Legate.default_alignment(task, Vector{Legate.Variable}(), init_output_vars)

    Legate.submit_task(rt, task)
    sleep(2)

    # 2. Compute Task (3 args)
    task2 = Legate.create_julia_task(rt, lib, my_task)
    input_vars = Vector{Legate.Variable}()
    output_vars = Vector{Legate.Variable}()
    push!(input_vars, Legate.add_input(task2, a))
    push!(input_vars, Legate.add_input(task2, b))
    push!(output_vars, Legate.add_output(task2, c))
    Legate.default_alignment(task2, input_vars, output_vars)

    Legate.submit_task(rt, task2)

    # 3. Arbitrary Arg Task (4 args: 2 in, 2 out)
    task3 = Legate.create_julia_task(rt, lib, my_4arg_task)
    in_vars_4 = Vector{Legate.Variable}()
    out_vars_4 = Vector{Legate.Variable}()
    push!(in_vars_4, Legate.add_input(task3, a))
    push!(in_vars_4, Legate.add_input(task3, c))
    push!(out_vars_4, Legate.add_output(task3, b))
    push!(out_vars_4, Legate.add_output(task3, d))
    Legate.default_alignment(task3, in_vars_4, out_vars_4)

    Legate.submit_task(rt, task3)

    # 4. Scalar Arg Task (2 args + scalar)
    task4 = Legate.create_julia_task(rt, lib, my_scalar_task)
    in_vars_s = Vector{Legate.Variable}()
    out_vars_s = Vector{Legate.Variable}()
    push!(in_vars_s, Legate.add_input(task4, c))
    push!(out_vars_s, Legate.add_output(task4, a))
    Legate.add_scalar(task4, Legate.Scalar(2.5f0))
    Legate.default_alignment(task4, in_vars_s, out_vars_s)

    Legate.submit_task(rt, task4)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_driver()
end
