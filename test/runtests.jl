using Legate
using Test
using HDF5

const VERBOSE = get(ENV, "VERBOSE", "1") != "0"
const run_gpu_tests =
    (get(ENV, "GPUTESTS", "1") != "0") && (get(ENV, "LEGATE_WRAPPER_ENABLE_CUDA", "ON") != "OFF")
@info "Run GPU Tests: $(run_gpu_tests)"

if run_gpu_tests
    using CUDA
    import CUDA: i32
    if CUDA.functional()
        VERBOSE && println(CUDA.versioninfo())
    else
        error("CUDA is not functional. GPU tests cannot be run.")
    end
end

include("tests/hdf5.jl")
include("tests/stability.jl")

# include("tests/tasking.jl")
# if run_gpu_tests
#     include("tests/tasking_gpu.jl")
# end
