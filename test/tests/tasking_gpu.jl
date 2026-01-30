function gpu_add_kernel(a, b, c)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(a)
        @inbounds c[idx] = a[idx] + b[idx]
    end
    return nothing
end

function init_cpu_task(args::Vector{Legate.TaskArgument})
    a, b = args
    fill!(a, 1.0f0)
    fill!(b, 2.0f0)
end

function read_cpu_task(args::Vector{Legate.TaskArgument})
    src, dest = args
    copyto!(dest, src)
end

@testset "GPU Tasking" begin
    rt = Legate.get_runtime()
    lib = Legate.create_library("gpu_test_lib")

    gpu_task_wrapped = Legate.wrap_task(gpu_add_kernel; task_type=:gpu)

    init_wrapped = Legate.wrap_task(init_cpu_task)
    # read_cpu_task unused in main, using closure below to capture result_host

    N = 100
    a = Legate.create_array([N], Float32)
    b = Legate.create_array([N], Float32)
    c = Legate.create_array([N], Float32)

    # 1. Initialize A and B (CPU)
    t1 = Legate.create_julia_task(rt, lib, init_wrapped)
    outs1 = [Legate.add_output(t1, a), Legate.add_output(t1, b)]
    Legate.default_alignment(t1, Vector{Legate.Variable}(), outs1)
    Legate.submit_task(rt, t1)

    # 2. Execute GPU Add (C = A + B)
    t2 = Legate.create_julia_task(rt, lib, gpu_task_wrapped)
    ins2 = [Legate.add_input(t2, a), Legate.add_input(t2, b)]
    outs2 = [Legate.add_output(t2, c)]
    Legate.default_alignment(t2, ins2, outs2)
    Legate.submit_task(rt, t2)

    # 3. Read Result (CPU copy)
    result_host = zeros(Float32, N)

    function read_task_closure(args::Vector{Legate.TaskArgument})
        src = args[1]
        for i in eachindex(src)
            result_host[i] = src[i]
        end
    end
    read_wrapped_closure = Legate.wrap_task(read_task_closure)

    t3 = Legate.create_julia_task(rt, lib, read_wrapped_closure)
    ins3 = [Legate.add_input(t3, c)]
    Legate.default_alignment(t3, ins3, Vector{Legate.Variable}())
    Legate.submit_task(rt, t3)

    Legate.wait_ufi()

    @test all(result_host .== 3.0f0)
end
