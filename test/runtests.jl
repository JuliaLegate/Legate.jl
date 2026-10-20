include("cuda_startup.jl")

using Legate
using Test
using HDF5
using Pkg

@testset "Realm backtrace configuration" begin
    withenv("REALM_BACKTRACE" => nothing) do
        @test Legate._configure_realm_backtrace!() == "0"
        @test ENV["REALM_BACKTRACE"] == "0"
    end
    withenv("REALM_BACKTRACE" => "1") do
        @test Legate._configure_realm_backtrace!() == "1"
        @test ENV["REALM_BACKTRACE"] == "1"
    end
end

@testset "UFI thread configuration" begin
    @test Legate._check_ufi_thread_configuration(2, :default) === nothing
    @test Legate._check_ufi_thread_configuration(1, :interactive) === nothing
    error = try
        Legate._check_ufi_thread_configuration(1, :default)
    catch exception
        exception
    end
    @test error isa ErrorException
    @test occursin("--threads=2", error.msg)
    @test occursin("--threads=1,1", error.msg)
end

@testset "CUDA artifact selection" begin
    if isdefined(Legate, :legate_jll)
        jll = Legate.legate_jll
        platform = deepcopy(jll.host_platform)
        artifacts = Pkg.Artifacts.find_artifacts_toml(pathof(jll))
        for (tag, expected) in (("13.4", "13.0"), ("12.9", nothing), ("none", "none"))
            platform["cuda"] = tag
            metadata = Pkg.Artifacts.artifact_meta("legate", artifacts; platform)
            @test (metadata === nothing ? nothing : metadata["cuda"]) == expected
        end
    end
end

const VERBOSE = get(ENV, "VERBOSE", "1") != "0"
const run_gpu_tests =
    (get(ENV, "GPUTESTS", "1") != "0") && (get(ENV, "LEGATE_WRAPPER_ENABLE_CUDA", "ON") != "OFF")
@info "Run GPU Tests: $(run_gpu_tests)"

if run_gpu_tests && !isnothing(Base.find_package("CUDA"))
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
include("tests/basic.jl")

include("tests/tasking/setup.jl")
include("tests/tasking/basic.jl")
include("tests/tasking/constraints.jl")
if run_gpu_tests
    include("tests/tasking/gpu.jl")
end
