module Legate
using Libdl

using CxxWrap

function preload_libs()
    libs = [
        joinpath(HDF5_ROOT, "lib", "libhdf5.so.310"),
        joinpath(NCCL_ROOT, "lib", "libnccl.so.2"),
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

    const LEGATE_ROOT = legate_jll.artifact_dir
    const LEGATE_WRAPPER_ROOT = legate_jl_wrapper_jll.artifact_dir
    const HDF5_ROOT = HDF5_jll.artifact_dir
    const NCCL_ROOT = NCCL_jll.artifact_dir
end

preload_libs() # for precompilation
@wrapmodule(() -> joinpath(LEGATE_WRAPPER_ROOT, "lib", "liblegate_jl_wrapper.so"))

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