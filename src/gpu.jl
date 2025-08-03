using CUDA
using CUDA_Driver_jll
using CUDA_Runtime_jll

function load_jll_lib(jll, lib)
    dir = joinpath(jll.artifact_dir, "lib")
    libpath = joinpath(dir, lib)
    try
        Libdl.dlopen(path, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    catch e
        @warn "Failed to open $(lib)" path=libpath exception=e
    end
    push!(Base.DL_LOAD_PATH, dir)
end

load_jll_lib(CUDA_Driver_jll, "libcuda.so")
load_jll_lib(CUDA_Runtime_jll, "libcudart.so")
CUDA.precompile_runtime()