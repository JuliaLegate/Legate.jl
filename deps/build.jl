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
using legate_jl_wrapper_jll
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


function build_cpp_wrapper(repo_root, legate_root, hdf5_root, nccl_root)
    @info "liblegatewrapper: Building C++ Wrapper Library"
    install_dir = joinpath(repo_root, "deps", "legate_wrapper_install")
    if isdir(install_dir)
        @warn "liblegatewrapper: Build dir exists. Deleting prior build."
        rm(install_dir, recursive = true)
        mkdir(install_dir)
    end

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()
    run_sh(`bash $build_cpp_wrapper $repo_root $legate_root $hdf5_root $nccl_root $install_dir $nthreads`, "cpp_wrapper")
end


function is_legate_installed(legate_dir::String; throw_errors::Bool = false)
    include_dir = joinpath(legate_dir, "include")
    if !isdir(joinpath(include_dir, "legate/legate"))
        throw_errors && @error "Legate.jl: Cannot find include/legate/legate in $(legate_dir)"
        return false
    end 
    return true
end


function parse_legate_version(legate_dir)
    version_file = joinpath(legate_dir, "include", "legate/legate", "version.h")

    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end-2])[end])
        minor = lpad(split(data[end-1])[end], 2, '0')
        patch = lpad(split(data[end])[end], 2, '0')
        version = "$(major).$(minor).$(patch)"
    end

    if isnothing(version)
        error("Legate.jl: Failed to parse version")
    end

    return version
end


function check_prefix_install(env_var, env_loc)
    if get(ENV, env_var, "0") == "1"
        @info "Legate.jl: Using $(env_var) mode"
        legate_dir = get(ENV, env_loc, nothing)
        legate_installed = is_legate_installed(legate_dir)
        if !legate_installed
            error("Legate.jl: Build halted: legate not found in $legate_dir")
        end
        installed_version = parse_legate_version(legate_dir)
        if installed_version ∉ SUPPORTED_LEGATE_VERSIONS
            error("Legate.jl: Build halted: $(legate_dir) detected unsupported version $(installed_version)")
        end
        @info "Legate.jl: Found a valid install in: $(legate_dir)"
        return true
    end
    return false
end


function build(run_legion_patch::Bool = true)
    pkg_root = abspath(joinpath(@__DIR__, "../"))
    deps_dir = joinpath(@__DIR__)

    @info "Legate.jl: Parsed Package Dir as: $(pkg_root)"
    hdf5_root = HDF5_jll.HDF5_jll.artifact_dir 
    nccl_root = NCCL_jll.NCCL_jll.artifact_dir

    # custom install
    if check_prefix_install("LEGATE_CUSTOM_INSTALL", "LEGATE_CUSTOM_INSTALL_LOCATION")
        legate_root = get(ENV, "LEGATE_CUSTOM_INSTALL_LOCATION", nothing)
    # conda install 
    elseif check_prefix_install("CUNUMERIC_LEGATE_CONDA_INSTALL", "CONDA_PREFIX")
        legate_root = get(ENV, "CONDA_PREFIX", nothing)
    else # default  
        legate_root = legate_jll.legate_jll.artifact_dir # the jll already has legate patched
        run_legion_patch = false
    end

    run_legion_patch && patch_legion(pkg_root, legate_root) # only patch if not legate_jll

    if get(ENV, "LEGATE_DEVELOP_MODE", "0") == "1"
        build_jlcxxwrap(pkg_root) # $pkg_root/libcxxwrap-julia 
        build_cpp_wrapper(pkg_root, legate_root, hdf5_root, nccl_root) # $pkg_root/wrapper
        legate_wrapper_root = joinpath(pkg_root, "deps", "legate_wrapper_install")
    else
        legate_wrapper_root = legate_jl_wrapper_jll.legate_jl_wrapper_jll.artifact_dir
    end

    # create lib_legatewrapper.so
    open(joinpath(deps_dir, "deps.jl"), "w") do io
        println(io, "const LEGATE_ROOT = \"$(legate_root)\"")
        println(io, "const LEGATE_WRAPPER_ROOT = \"$(legate_wrapper_root)\"")
        println(io, "const HDF5_ROOT = \"$(hdf5_root)\"")
        println(io, "const NCCL_ROOT = \"$(nccl_root)\"")
    end 
end

build()