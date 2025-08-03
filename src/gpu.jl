using CUDA
using CUDA_Driver_jll

# Load libcuda globally
const cuda_driver_dir = joinpath(CUDA_Driver_jll.artifact_dir, "lib")
const libcuda_path = joinpath(cuda_driver_dir, "libcuda.so")
try
    Libdl.dlopen(libcuda_path, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
catch e
    @warn "Failed to open libcuda.so" path=libcuda_path exception=e
end
push!(Base.DL_LOAD_PATH, cuda_driver_dir)

CUDA.precompile_runtime()