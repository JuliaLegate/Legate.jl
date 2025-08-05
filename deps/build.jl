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
using Preferences
import LegatePreferences

# Automatically pipes errors to new file
# and appends stdout to build.log
function run_sh(cmd::Cmd, filename::String)
    println(cmd)

    build_log = joinpath(@__DIR__, "build.log")
    tmp_build_log = joinpath(@__DIR__, "$(filename).log")
    err_log = joinpath(@__DIR__, "$(filename).err")

    if isfile(err_log)
        rm(err_log)
    end

    if isfile(tmp_build_log)
        rm(tmp_build_log)
    end

    try
        run(pipeline(cmd; stdout=tmp_build_log, stderr=err_log, append=false))
        println(contents) 
        contents = read(tmp_build_log, String)
        open(build_log, "a") do io
            println(contents)
        end
    catch e
        println("stderr log generated: ", err_log, '\n')
        contents = read(err_log, String)
        if !isempty(strip(contents))
            println("---- Begin stderr log ----")
            println(contents)
            println("---- End stderr log ----")
        end
    end
end

# patch legion. The readme below talks about our compilation error
# https://github.com/ejmeitz/cuNumeric.jl/blob/main/scripts/README.md
function patch_legion(repo_root::String, legate_root::String)
    @info "Legate.jl: Patching Legion"
    
    legion_patch = joinpath(repo_root, "scripts/patch_legion.sh")
    @info "Legate.jl: Running legion patch script: $legion_patch"
    run_sh(`bash $legion_patch $repo_root $legate_root`, "legion_patch")
end

function build_jlcxxwrap(repo_root)
    @info "libcxxwrap: Downloading"
    build_libcxxwrap = joinpath(repo_root, "scripts/install_cxxwrap.sh")
    
    version_path = joinpath(DEPOT_PATH[1], "dev/libcxxwrap_julia_jll/override/LEGATE_INSTALL.txt")

    if isfile(version_path)
        version = strip(read(version_path, String))
        if version ∈ SUPPORTED_LEGATE_VERSIONS
            @info "libcxxwrap: Found supported version built with Legate.jl: $version"
            return
        else
            @info "libcxxwrap: Unsupported version found: $version. Rebuilding..."
        end
    else
        @info "libcxxwrap: No version file found. Starting build..."
    end
  
    @info "libcxxwrap: Running build script: $build_libcxxwrap"
    run_sh(`bash $build_libcxxwrap $repo_root`, "libcxxwrap")
    open(version_path, "w") do io
        write(io, LATEST_LEGATE_VERSION)
    end
end


function build_cpp_wrapper(repo_root, legate_root, hdf5_root, nccl_root, install_dir)
    @info "liblegatewrapper: Building C++ Wrapper Library"
    if isdir(install_dir)
        @warn "liblegatewrapper: Build dir exists. Deleting prior build."
        rm(install_dir, recursive = true)
        mkdir(install_dir)
    end
    branch = load_preference("Legate", "wrapper_branch", LegatePreferences.DEVEL_DEFAULT_WRAPPER_BRANCH)

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()
    run_sh(`bash $build_cpp_wrapper $repo_root $legate_root $hdf5_root $nccl_root $install_dir $branch $nthreads`, "cpp_wrapper")
end

function build(mode)
    if mode == LegatePreferences.MODE_JLL
        @warn "No reason to Build on JLL mode."
    end

    pkg_root = abspath(joinpath(@__DIR__, "../"))
    deps_dir = joinpath(@__DIR__)

    build_log = joinpath(deps_dir, "build.log")
    open(build_log, "w") do io
        println(io, "=== Build started ===")
    end

    @info "Legate.jl: Parsed Package Dir as: $(pkg_root)"

    hdf5_lib = load_preference("Legate", "HDF5_LIB", nothing)
    hdf5_root = joinpath(hdf5_lib, "..")
    nccl_lib = load_preference("Legate", "NCCL_LIB", nothing)
    nccl_root = joinpath(nccl_lib, "..")

    # only patch if not legate_jll
    if mode == LegatePreferences.MODE_DEVELOPER || mode == LegatePreferences.MODE_CONDA
        patch_legion(pkg_root, legate_root) 
    end
    
    if mode == LegatePreferences.MODE_DEVELOPER
        install_dir = load_preference("Legate", "LEGATE_WRAPPER_LIB", nothing)
        build_jlcxxwrap(pkg_root) # $pkg_root/libcxxwrap-julia 
        build_cpp_wrapper(pkg_root, legate_root, hdf5_root, nccl_root, install_dir) # $pkg_root/wrapper
    end
end

const JULIA_LEGATE_BUILDING_DOCS = get(ENV, "JULIA_LEGATE_BUILDING_DOCS", "false") == "true"
const mode = load_preference("Legate", "mode", LegatePreferences.MODE_JLL)

if !JULIA_LEGATE_BUILDING_DOCS
    build(mode)
end