# GPU kernels receive their operands as a single tuple (see _gpu_launch).
function gpu_add_kernel(args)
    a, b, c = args
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(a)
        @inbounds c[idx] = a[idx] + b[idx]
    end
    return nothing
end

function init_cpu_task(a, b)
    fill!(a, 1.0f0)
    return fill!(b, 2.0f0)
end

# 2D, layout-sensitive kernel: exercises strided-tile gather/scatter under partitioning.
function gpu_scale2d_kernel(args)
    a, c = args
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if ix <= length(a)
        m = size(a, 1)
        i = (ix - 1) % m + 1
        j = (ix - 1) ÷ m + 1
        @inbounds c[i, j] = a[i, j] * 2.0f0
    end
    return nothing
end

@testset "GPU Tasking" begin
    Legate.Experimental(true)  # tasking is experimental
    rt = Legate.get_runtime()
    lib = Legate.create_library("gpu_test_lib")

    gpu_task_wrapped = Legate.wrap_task(gpu_add_kernel, Legate.GPUBackend)
    init_wrapped = Legate.wrap_task(init_cpu_task, Legate.CPUBackend)

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
    result_host = Array(c)

    @test all(result_host .== 3.0f0)

    # 2D layout-sensitive task over strided tiles (partitioned under LEGATE_TEST=1).
    scale_wrapped = Legate.wrap_task(gpu_scale2d_kernel, Legate.GPUBackend)
    M, K = 4, 8
    pattern = Float32[i + 100 * j for i in 1:M, j in 1:K]
    a2 = Legate.create_array([M, K], Float32)
    c2 = Legate.create_array([M, K], Float32)
    copyto!(a2, pattern)

    t3 = Legate.create_julia_task(rt, lib, scale_wrapped)
    ins3 = [Legate.add_input(t3, a2)]
    outs3 = [Legate.add_output(t3, c2)]
    Legate.default_alignment(t3, ins3, outs3)
    Legate.submit_task(rt, t3)

    @test Array(c2) ≈ 2 .* pattern
end
