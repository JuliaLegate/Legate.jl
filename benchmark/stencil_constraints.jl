using Legate

const BACKEND = lowercase(get(ENV, "STENCIL_BACKEND", "cpu"))
const GRID_SIZE = parse(Int, get(ENV, "STENCIL_SIZE", "1024"))
const ITERATIONS = parse(Int, get(ENV, "STENCIL_ITERATIONS", "100"))
const WARMUP_ITERATIONS = parse(Int, get(ENV, "STENCIL_WARMUP", "5"))
const SAMPLES = parse(Int, get(ENV, "STENCIL_SAMPLES", "5"))

if BACKEND == "gpu"
    @eval using CUDA
elseif BACKEND != "cpu"
    error("STENCIL_BACKEND must be cpu or gpu")
end

# The field stays fixed so timing isolates repeated partitioning and materialization.
function aligned_stencil(north, south, west, east, center, output)
    @inbounds @simd for i in eachindex(output)
        output[i] =
            0.2f0 * (north[i] + south[i] + west[i] + east[i] + center[i])
    end
    return nothing
end

function bloated_stencil(core, halo, output)
    n = GRID_SIZE
    # UFI arrays are locally indexed; monotonic values reveal the clipped halo offset.
    delta = round(Int, core[1, 1] - halo[1, 1])
    low_i = delta % n
    low_j = delta ÷ n
    @inbounds for j in axes(output, 2), i in axes(output, 1)
        hi = i + low_i
        hj = j + low_j
        north = hi > 1 ? halo[hi - 1, hj] : 0.0f0
        south = hi < size(halo, 1) ? halo[hi + 1, hj] : 0.0f0
        west = hj > 1 ? halo[hi, hj - 1] : 0.0f0
        east = hj < size(halo, 2) ? halo[hi, hj + 1] : 0.0f0
        output[i, j] =
            0.2f0 * (north + south + west + east + core[i, j])
    end
    return nothing
end

function aligned_stencil_gpu(args)
    north, south, west, east, center, output = args
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= length(output)
        @inbounds output[index] =
            0.2f0 *
            (north[index] + south[index] + west[index] + east[index] + center[index])
    end
    return nothing
end

function bloated_stencil_gpu(args)
    core, halo, output = args
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= length(output)
        n = GRID_SIZE
        delta = round(Int, core[1, 1] - halo[1, 1])
        low_i = delta % n
        low_j = delta ÷ n
        i = (index - 1) % size(output, 1) + 1
        j = (index - 1) ÷ size(output, 1) + 1
        hi = i + low_i
        hj = j + low_j
        north = hi > 1 ? halo[hi - 1, hj] : 0.0f0
        south = hi < size(halo, 1) ? halo[hi + 1, hj] : 0.0f0
        west = hj > 1 ? halo[hi, hj - 1] : 0.0f0
        east = hj < size(halo, 2) ? halo[hi, hj + 1] : 0.0f0
        @inbounds output[i, j] =
            0.2f0 * (north + south + west + east + core[i, j])
    end
    return nothing
end

function region(array, first, last)
    view = Legate.slice(array, 0, first[1], last[1])
    return Legate.slice(view, 1, first[2], last[2])
end

function stencil_views(array, n)
    center = region(array, (1, 1), (n + 1, n + 1))
    north = region(array, (0, 1), (n, n + 1))
    south = region(array, (2, 1), (n + 2, n + 1))
    west = region(array, (1, 0), (n + 1, n))
    east = region(array, (1, 2), (n + 1, n + 2))
    return (; north, south, west, east, center)
end

function make_aligned_problem(n)
    input_host = zeros(Float32, n + 2, n + 2)
    @inbounds for j in 1:n, i in 1:n
        input_host[i + 1, j + 1] = Float32(i + n * (j - 1))
    end

    input = Legate.create_array([n + 2, n + 2], Float32)
    output = Legate.create_array([n, n], Float32)
    copyto!(input, input_host)
    views = stencil_views(input, n)
    return (; input, output, views)
end

function make_bloated_problem(n)
    input_host = zeros(Float32, n + 2, n + 2)
    @inbounds for j in 1:n, i in 1:n
        input_host[i + 1, j + 1] = Float32(i + n * (j - 1))
    end

    input = Legate.create_array([n + 2, n + 2], Float32)
    output = Legate.create_array([n, n], Float32)
    copyto!(input, input_host)
    core = region(input, (1, 1), (n + 1, n + 1))
    halo = region(input, (1, 1), (n + 1, n + 1))
    return (; input, core, halo, output)
end

function submit_aligned!(runtime, library, wrapped, problem)
    task = Legate.create_julia_task(runtime, library, wrapped)
    inputs = [
        Legate.add_input(task, problem.views.north),
        Legate.add_input(task, problem.views.south),
        Legate.add_input(task, problem.views.west),
        Legate.add_input(task, problem.views.east),
        Legate.add_input(task, problem.views.center),
    ]
    outputs = [Legate.add_output(task, problem.output)]
    Legate.default_alignment(task, inputs, outputs)
    Legate.submit_task(runtime, task)
    return problem
end

function submit_bloated!(runtime, library, wrapped, problem)
    task = Legate.create_julia_task(runtime, library, wrapped)
    core = Legate.add_input(task, problem.core)
    halo = Legate.add_input(task, problem.halo)
    output = Legate.add_output(task, problem.output)
    Legate.add_constraint(task, Legate.align(core, output))
    Legate.add_constraint(task, Legate.bloat(core, halo, (1, 1), (1, 1)))
    Legate.submit_task(runtime, task)
    return problem
end

function synchronize()
    Legate.issue_execution_fence()
    Legate.wait_ufi()
    return nothing
end

function run_iterations!(submit!, runtime, library, wrapped, problem, iterations)
    for _ in 1:iterations
        submit!(runtime, library, wrapped, problem)
    end
    return problem
end

function elapsed_sample!(submit!, runtime, library, wrapped, problem, iterations)
    synchronize()
    start = time_ns()
    run_iterations!(submit!, runtime, library, wrapped, problem, iterations)
    synchronize()
    return (time_ns() - start) / 1.0e9
end

function median_value(values)
    sorted = sort(values)
    middle = length(sorted) ÷ 2
    isodd(length(sorted)) && return sorted[middle + 1]
    return (sorted[middle] + sorted[middle + 1]) / 2
end

function main()
    GRID_SIZE > 0 || error("STENCIL_SIZE must be positive")
    ITERATIONS > 0 || error("STENCIL_ITERATIONS must be positive")
    SAMPLES > 0 || error("STENCIL_SAMPLES must be positive")

    Legate.Experimental(true)
    runtime = Legate.get_runtime()
    library = Legate.create_library("stencil_constraint_benchmark")
    if BACKEND == "gpu"
        aligned = Legate.wrap_task(aligned_stencil_gpu, Legate.GPUBackend)
        bloated = Legate.wrap_task(bloated_stencil_gpu, Legate.GPUBackend)
    else
        aligned = Legate.wrap_task(aligned_stencil, Legate.CPUBackend)
        bloated = Legate.wrap_task(bloated_stencil, Legate.CPUBackend)
    end

    aligned_problem = make_aligned_problem(GRID_SIZE)
    bloated_problem = make_bloated_problem(GRID_SIZE)

    run_iterations!(
        submit_aligned!, runtime, library, aligned, aligned_problem, WARMUP_ITERATIONS
    )
    run_iterations!(
        submit_bloated!, runtime, library, bloated, bloated_problem, WARMUP_ITERATIONS
    )
    synchronize()

    aligned_times = Float64[]
    bloated_times = Float64[]
    for sample in 1:SAMPLES
        if isodd(sample)
            elapsed = elapsed_sample!(
                submit_aligned!, runtime, library, aligned, aligned_problem, ITERATIONS
            )
            push!(aligned_times, elapsed)
            elapsed = elapsed_sample!(
                submit_bloated!, runtime, library, bloated, bloated_problem, ITERATIONS
            )
            push!(bloated_times, elapsed)
        else
            elapsed = elapsed_sample!(
                submit_bloated!, runtime, library, bloated, bloated_problem, ITERATIONS
            )
            push!(bloated_times, elapsed)
            elapsed = elapsed_sample!(
                submit_aligned!, runtime, library, aligned, aligned_problem, ITERATIONS
            )
            push!(aligned_times, elapsed)
        end
    end

    aligned_result = Array(aligned_problem.output)
    bloated_result = Array(bloated_problem.output)
    aligned_result == bloated_result || error("alignment and bloat results differ")

    aligned_median = median_value(aligned_times)
    bloated_median = median_value(bloated_times)
    println("5-point stencil constraint benchmark")
    println("  backend: $BACKEND")
    println("  grid: $(GRID_SIZE) x $(GRID_SIZE)")
    println("  iterations/sample: $ITERATIONS, samples: $SAMPLES")
    println("  default alignment median: $(round(aligned_median; digits=4)) s")
    println("  bloat median:             $(round(bloated_median; digits=4)) s")
    println("  bloat speedup:            $(round(aligned_median / bloated_median; digits=3))x")
    println("  alignment samples: $aligned_times")
    return println("  bloat samples:     $bloated_times")
end

main()
