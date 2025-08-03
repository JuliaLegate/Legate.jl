#= Copyright 2025 Northwestern University, 
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

module Legate
import Base: get

using OpenSSL_jll # Libdl requires OpenSSL 
using Libdl
using CxxWrap
using libaec_jll # must load prior to HDF5

include("gpu.jl")

function preload_libs()
    libs = [
        libaec_jll.get_libsz_path(), # required for libhdf5.so
        joinpath(MPI_LIB, "libmpicxx.so"), # required for libmpi.so
        joinpath(MPI_LIB, "libmpi.so"),   # legate_jll is configured with NCCL which requires MPI for CPU tasks
        joinpath(NCCL_LIB, "libnccl.so"), # legate_jll is configured with NCCL
        joinpath(HDF5_LIB, "libhdf5.so"), # legate_jll is configured with HDF5
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
    using HDF5_jll
    using NCCL_jll
    using MPICH_jll
    using legate_jll
    using legate_jl_wrapper_jll

    const HDF5_LIB = joinpath(HDF5_jll.artifact_dir, "lib")
    const NCCL_LIB = joinpath(NCCL_jll.artifact_dir, "lib")
    const MPI_LIB  = joinpath(MPICH_jll.artifact_dir, "lib")
    const LEGATE_LIB = joinpath(legate_jll.artifact_dir, "lib")
    const LEGATE_WRAPPER_LIB = joinpath(legate_jl_wrapper_jll.artifact_dir, "lib")
end

preload_libs() # for precompilation
@wrapmodule(() -> joinpath(LEGATE_WRAPPER_LIB, "liblegate_jl_wrapper.so"))
include("type.jl")


function my_on_exit()
    Legate.legate_finish()
end

function __init__()
    preload_libs() # for runtime
    @initcxx

    Legate.start_legate()
    @debug "Started Legate"
    @warn " Leagte.jl and cuNumeric.jl are under active development at the moment. This is a pre-release API and is subject to change. \
            Stability is not guaranteed until the first official release. We are actively working to improve the build experience to be \
            more seamless and Julia-friendly. In parallel, we're developing a comprehensive testing framework to ensure reliability and \
            robustness. Our public beta launch is targeted for Fall 2025. \
            If you are seeing this warning, I am impressed that you have successfully installed Legate.jl. \
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