using Legate
using Test

const VERBOSE = get(ENV, "VERBOSE", "1") != "0"
const run_gpu_tests = (get(ENV, "GPUTESTS", "1") != "0") && (get(ENV, "NO_CUDA", "OFF") != "ON")
@info "Run GPU Tests: $(run_gpu_tests)"

if run_gpu_tests && !isnothing(Base.find_package("CUDA"))
    using CUDA
    if CUDA.functional()
        import CUDA: i32
        VERBOSE && println(CUDA.versioninfo())
    else
        run_gpu_tests = false
    end
elseif run_gpu_tests
    run_gpu_tests = false # No CUDA.jl so we can't run CUDA tests
end

include("tests/tasking.jl")

if run_gpu_tests
    include("tests/tasking_gpu.jl")
end
