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

using Pkg
import Base: notnothing

using legate_jll
using NCCL_jll
using HDF5_jll

const SUPPORTED_LEGATE_VERSIONS = ["25.05.00"]
const LATEST_LEGATE_VERSION = SUPPORTED_LEGATE_VERSIONS[end]

# Automatically pipes errors to new file
# and appends stdout to build.log
function run_sh(cmd::Cmd, filename::String)

    println(cmd)
    
    build_log = joinpath(@__DIR__, "build.log")
    err_log = joinpath(@__DIR__, "$(filename).err")

    if isfile(err_log)
        rm(err_log)
    end

    try
        run(pipeline(cmd, stdout = build_log, stderr = err_log, append = false))
    catch e
        println("stderr log generated: ", err_log, '\n')
        exit(-1)
    end

end

# patch legion. The readme below talks about our compilation error
# https://github.com/ejmeitz/cuNumeric.jl/blob/main/scripts/README.md
function patch_legion(repo_root::String, legate_jll_root::String)
    @info "Patching Legion"
    
    legion_patch = joinpath(repo_root, "scripts/patch_legion.sh")
    @info "Running legion patch script: $legion_patch"
    run_sh(`bash $legion_patch $repo_root $legate_jll_root`, "legion_patch")
end

function build_jlcxxwrap(repo_root)
    @info "Downloading libcxxwrap"
    build_libcxxwrap = joinpath(repo_root, "scripts/install_cxxwrap.sh")
    
    version_path = joinpath(DEPOT_PATH[1], "dev/libcxxwrap_julia_jll/override/LEGATE_INSTALL.txt")

    if isfile(version_path)
        version = strip(read(version_path, String))
        if version ∈ SUPPORTED_LEGATE_VERSIONS
            @info "Found supported version: $version"
            return
        else
            @info "Unsupported version found: $version. Rebuilding..."
        end
    else
        @info "No version file found. Starting build..."
    end
  
    @info "Running libcxxwrap build script: $build_libcxxwrap"
    run_sh(`bash $build_libcxxwrap $repo_root`, "libcxxwrap")
    open(version_path, "w") do io
        write(io, LATEST_LEGATE_VERSION)
    end
end


function build_cpp_wrapper(repo_root, legate_jll_root, hdf5_jll_root, nccl_jll_root)
    @info "Building C++ Wrapper Library"
    build_dir = joinpath(repo_root, "wrapper", "build")
    if !isdir(build_dir)
        mkdir(build_dir)
    else
        @warn "Build dir exists. Deleting prior build."
        rm(build_dir, recursive = true)
        mkdir(build_dir)
    end

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()
    run_sh(`bash $build_cpp_wrapper $repo_root $legate_jll_root $hdf5_jll_root $nccl_jll_root $build_dir $nthreads`, "cpp_wrapper")
end


function build(run_legion_patch::Bool = true)
    pkg_root = abspath(joinpath(@__DIR__, "../"))
    @info "Parsed Package Dir as: $(pkg_root)"

    legate_jll_root = legate_jll.artifact_dir
    hdf5_jll_root = HDF5_jll.artifact_dir
    nccl_jll_root = NCCL_jll.artifact_dir

    run_legion_patch && patch_legion(pkg_root, legate_jll_root)

    # We still need to build libcxxwrap from source until 
    # everything is on BinaryBuilder to ensure compiler compatability
    build_jlcxxwrap(pkg_root)

    # create lib_legatewrapper.so
    build_cpp_wrapper(pkg_root, legate_jll_root, hdf5_jll_root, nccl_jll_root)
end



build()