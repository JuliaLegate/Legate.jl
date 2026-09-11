include("cuda_startup.jl")

using Legate
using Test
using HDF5

@testset "CUDA artifact selection" begin
    if isdefined(Legate, :legate_jll)
        jll = Legate.legate_jll
        platform = deepcopy(jll.host_platform)
        artifacts = Legate.Pkg.Artifacts.find_artifacts_toml(pathof(jll))
        for (tag, expected) in (("13.4", "13.0"), ("12.9", nothing), ("none", "none"))
            platform["cuda"] = tag
            metadata = Legate.Pkg.Artifacts.artifact_meta("legate", artifacts; platform)
            @test (metadata === nothing ? nothing : metadata["cuda"]) == expected
        end
    end
end

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
