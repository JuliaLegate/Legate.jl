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

using Preferences
import LegatePreferences

using OpenSSL_jll # Libdl requires OpenSSL 
using Libdl
using CxxWrap
using Hwloc_jll # needed for mpi 
using libaec_jll # must load prior to HDF5

using HDF5_jll
using MPICH_jll
using NCCL_jll
using legate_jll
using legate_jl_wrapper_jll # the wrapper depends on HDF5, MPICH, NCCL, and legate

using Pkg
using TOML 

const SUPPORTED_LEGATE_VERSIONS = ["25.05.00"]

function preload_libs()
    libs = [
        libaec_jll.get_libsz_path(), # required for libhdf5.so
        joinpath(Hwloc_jll.artifact_dir, "lib", "libhwloc.so"), # required for libmpicxx.so
        joinpath(MPI_LIB, "libmpicxx.so"), # required for libmpi.so
        joinpath(MPI_LIB, "libmpi.so"),   # legate_jll is configured with NCCL which requires MPI for CPU tasks
        joinpath(CUDA_RUNTIME_LIB, "libcudart.so"), # needed for libnccl.so and liblegate.so
        joinpath(CUDA_DRIVER_LIB, "libcuda.so"), # needed for liblegate.so
        joinpath(NCCL_LIB, "libnccl.so"), # legate_jll is configured with NCCL
        joinpath(HDF5_LIB, "libhdf5.so"), # legate_jll is configured with HDF5
        joinpath(LEGATE_LIB, "liblegate.so"), 
    ]
    for lib in libs
        Libdl.dlopen(lib, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    end
end

using CUDA
using CUDA_Driver_jll
using CUDA_Runtime_jll
include("preference.jl")
find_preferences()

const MPI_LIB = load_preference(LegatePreferences, "MPI_LIB", nothing)
const CUDA_RUNTIME_LIB = load_preference(LegatePreferences, "CUDA_RUNTIME_LIB", nothing)
const CUDA_DRIVER_LIB = load_preference(LegatePreferences, "CUDA_DRIVER_LIB", nothing)
const NCCL_LIB = load_preference(LegatePreferences, "NCCL_LIB", nothing)
const HDF5_LIB = load_preference(LegatePreferences, "HDF5_LIB", nothing)
const LEGATE_LIB = load_preference(LegatePreferences, "LEGATE_LIB", nothing)
const LEGATE_WRAPPER_LIB = load_preference(LegatePreferences, "LEGATE_WRAPPER_LIB", nothing)

libpath = joinpath(LEGATE_WRAPPER_LIB, "liblegate_jl_wrapper.so")

preload_libs() # for precompilation
@wrapmodule(() -> libpath)

include("util.jl")
include("type.jl")

function my_on_exit()
    Legate.legate_finish()
end

function __init__()
    LegatePreferences.check_unchanged()
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
end 