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

const SUPPORTED_LEGATE_VERSIONS = ["25.05.00"]
const LATEST_LEGATE_VERSION = SUPPORTED_LEGATE_VERSIONS[end]

up_dir(dir::String) = abspath(joinpath(dir, ".."))

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


function build_cpp_wrapper(repo_root, legate_root, hdf5_root, nccl_root, install_root)
    @info "liblegatewrapper: Building C++ Wrapper Library"
    if isdir(install_root)
        rm(install_root, recursive = true)
        mkdir(install_root)
    end
    branch = load_preference(LegatePreferences, "wrapper_branch", LegatePreferences.DEVEL_DEFAULT_WRAPPER_BRANCH)

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()
    run_sh(`bash $build_cpp_wrapper $repo_root $legate_root $hdf5_root $nccl_root $install_root $branch $nthreads`, "cpp_wrapper")
end

function replace_nothing_jll(lib, jll)
    if isnothing(lib)
        eval(:(using $(jll)))
        jll_mod = getfield(Main, jll)
        lib = joinpath(jll_mod.artifact_dir, "lib")
    end
    return lib
end

function replace_nothing_conda_jll(mode, lib, jll)
    if isnothing(lib)
        if mode == LegatePreferences.MODE_CONDA
            lib = up_dir(load_preference(LegatePreferences, "conda_env", nothing))
        else
            eval(:(using $(jll)))
            jll_mod = getfield(Main, jll)
            lib = joinpath(jll_mod.artifact_dir, "lib")
        end
    end
    return lib
end

function build(mode)
    if mode == LegatePreferences.MODE_JLL
        @warn "No reason to Build on JLL mode. Exiting Build"
        return
    end

    pkg_root = up_dir(@__DIR__)
    deps_dir = joinpath(@__DIR__)

    build_log = joinpath(deps_dir, "build.log")
    open(build_log, "w") do io
        println(io, "=== Build started ===")
    end

    @info "Legate.jl: Parsed Package Dir as: $(pkg_root)"

    hdf5_lib = load_preference(LegatePreferences, "HDF5_LIB", nothing)
    nccl_lib = load_preference(LegatePreferences, "NCCL_LIB", nothing)
    legate_lib = load_preference(LegatePreferences, "LEGATE_LIB", nothing)

    hdf5_lib   = replace_nothing_jll(hdf5_lib, :HDF5_jll)
    nccl_lib   = replace_nothing_conda_jll(mode, nccl_lib, :NCCL_jll)
    legate_lib = replace_nothing_conda_jll(mode, legate_lib, :legate_jll)

    # only patch if not legate_jll
    if mode == LegatePreferences.MODE_DEVELOPER || mode == LegatePreferences.MODE_CONDA
        patch_legion(pkg_root, up_dir(legate_lib)) 
    end
    
    if mode == LegatePreferences.MODE_DEVELOPER
        install_dir = joinpath(pkg_root, "deps", "legate_jl_wrapper")
        build_jlcxxwrap(pkg_root) # $pkg_root/libcxxwrap-julia 
        # build_cpp_wrapper wants roots of every library
        build_cpp_wrapper(pkg_root, up_dir(legate_lib), up_dir(hdf5_lib), up_dir(nccl_lib), install_dir) # $pkg_root/legate_jl_wrapper
    end
end

const mode = load_preference(LegatePreferences, "mode", LegatePreferences.MODE_JLL)
build(mode)