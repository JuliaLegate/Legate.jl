module Legate
using OpenSSL_jll
using libaec_jll
using CUDA_Driver_jll
using Libdl
using CxxWrap

function preload_libs()
    libs = [
        libaec_jll.get_libsz_path(),
        joinpath(CUDA_Driver_jll.artifact_dir, "lib", "libcuda.so"),
        joinpath(MPI_LIB, "libmpicxx.so"),
        joinpath(MPI_LIB, "libmpi.so"),
        joinpath(NCCL_LIB, "libnccl.so"),
        joinpath(HDF5_LIB, "libhdf5.so"),
        joinpath(LEGATE_LIB, "liblegate.so"), 
    ]
    for lib in libs
        Libdl.dlopen(lib, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    end
end 

deps_path = joinpath(@__DIR__, "../deps/deps.jl")

if isfile(deps_path)
    # deps.jl should assign to the Refs, not declare new consts
    include(deps_path)
else
    using legate_jll
    using legate_jl_wrapper_jll
    using HDF5_jll
    using NCCL_jll
    using MPICH_jll

    const LEGATE_LIB = joinpath(legate_jll.artifact_dir, "lib")
    const LEGATE_WRAPPER_LIB = joinpath(legate_jl_wrapper_jll.artifact_dir, "lib")
    const HDF5_LIB = joinpath(HDF5_jll.artifact_dir, "lib")
    const NCCL_LIB = joinpath(NCCL_jll.artifact_dir, "lib")
    const MPI_LIB  = joinpath(MPICH_jll.artifact_dir, "lib")
end

preload_libs() # for precompilation
@wrapmodule(() -> joinpath(LEGATE_WRAPPER_LIB, "liblegate_jl_wrapper.so"))

include("type.jl")

function my_on_exit()
    @debug "Cleaning Up Legate"
    Legate.legate_finish()
end

function __init__()
    preload_libs() # for runtime
    @initcxx

    Legate.start_legate()
    @debug "Started Legate"
    @warn "Leagte.jl and cuNumeric.jl are under active development at the moment and may change its API and supported end systems at any time. \
           If you are seeing this warning, I am impressed that you have successfully installed Legate.jl. We are working to make the build \
           experience Julia much more Julia friendly. We are also working to create exhaustive testing. Public beta launch aimed for Fall 2025. \
    "
    Base.atexit(my_on_exit)
end

function get_install_liblegate()
    return LEGATE_LIB
end

function get_install_libnccl()
    return NCCL_LIB
end

function get_install_libmpi()
    return MPI_LIB
end

function get_install_libhdf5()
    return HDF5_LIB
end

end 