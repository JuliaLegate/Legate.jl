module Legate
using Libdl

using CxxWrap

function preload_libs()
    libs = [
        joinpath(HDF5_ROOT, "libhdf5.so.310"),
        joinpath(NCCL_ROOT, "libnccl.so.2"),
        joinpath(LEGATE_ROOT, "liblegate.so.25.05.00"), 
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

    const LEGATE_ROOT = joinpath(legate_jll.artifact_dir, "lib")
    const LEGATE_WRAPPER_ROOT = joinpath(legate_jl_wrapper_jll.artifact_dir, "lib")
    const HDF5_ROOT = joinpath(HDF5_jll.artifact_dir, "lib")
    const NCCL_ROOT = joinpath(NCCL_jll.artifact_dir, "lib")

preload_libs() # for precompilation
@wrapmodule(() -> joinpath(LEGATE_WRAPPER_ROOT, "liblegate_jl_wrapper.so"))

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
    return LEGATE_ROOT
end

end 