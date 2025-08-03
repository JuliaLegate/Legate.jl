module GPUSetup

# Ensure CUDA and CUDA_Driver_jll are loaded into Main (global)
if !isdefined(Main, :CUDA)
    Core.eval(Main, :(using CUDA))
end

if !isdefined(Main, :CUDA_Driver_jll)
    Core.eval(Main, :(using CUDA_Driver_jll))
end

# Load libcuda globally
const cuda_driver_dir = joinpath(CUDA_Driver_jll.artifact_dir, "lib")
const libcuda_path = joinpath(cuda_driver_dir, "libcuda.so")
try
    Libdl.dlopen(libcuda_path, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
catch e
    @warn "Failed to open libcuda.so" path=libcuda_path exception=e
end

ENV["LD_LIBRARY_PATH"] = cuda_driver_dir * ":" * get(ENV, "LD_LIBRARY_PATH", "")
CUDA.precompile_runtime()
end # module
