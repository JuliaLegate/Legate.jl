# GPU-independent checks; CUDA.jl itself owns driver initialization and discovery.
module CUDAStartupTests
using Test
module CUDACore
const available = Ref(true)
const version = Ref(v"13.0")
has_cuda_gpu(show_reason=false) = available[]
driver_version() = version[]
end
module legate_jll
const host_platform = (tags=Dict("cuda" => "13.4"),)
end
abstract type Mode end
struct JLL <: Mode end
struct Developer <: Mode end
struct Conda <: Mode end
const LegatePreferences = nothing
const MIN_CUDA_VERSION = v"13.0"
load_preference(args...) = false
include("../src/utilities/cuda.jl")

@testset "CUDA startup check" begin
    @test has_cuda_gpu()
    @test _check_cuda_version(v"13.0") === nothing
    @test _check_cuda_version(v"14.0") === nothing
    @test_throws ErrorException _check_cuda_version(v"12.9")
    @test _cpu_only_config("--gpus 0 --cpus 1")
    @test _cpu_only_config("--gpus=0")
    @test !_cpu_only_config("--gpus 0 --gpus 1")
    @test !_cpu_only_config("")
    @test _check_cuda(Conda()) === nothing
    @test _check_cuda(Developer()) === nothing
    withenv("LEGATE_CONFIG" => nothing, "LEGATE_AUTO_CONFIG" => nothing) do
        @test !_cpu_only_config()
        @test _check_cuda(JLL()) === nothing
        CUDACore.version[] = v"12.9"
        @test_throws ErrorException _check_cuda(JLL())
        CUDACore.version[] = v"13.0"
    end
    withenv("LEGATE_CONFIG" => "--gpus 1") do
        @test _check_cuda(JLL()) === nothing # toolkit 13.4 does not require driver 13.4
        CUDACore.version[] = v"12.9"
        @test_throws ErrorException _check_cuda(JLL())
        CUDACore.available[] = false
        @test !has_cuda_gpu()
        @test_throws ErrorException _check_cuda(JLL())
        legate_jll.host_platform.tags["cuda"] = "none"
        @test _check_cuda(JLL()) === nothing
        legate_jll.host_platform.tags["cuda"] = "13.4"
        withenv("LEGATE_CONFIG" => "--gpus 0") do
            @test _check_cuda(JLL()) === nothing
        end
    end
end
end
