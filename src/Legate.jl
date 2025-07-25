module Legate
using OpenSSL_jll
using Libdl
using CxxWrap

function preload_libs()
    libs = [
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
    @info "Cleaning Up Legate"
    Legate.legate_finish()
end

function __init__()
    preload_libs() # for runtime
    @initcxx

    Legate.start_legate()
    @info "Started Legate"
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