#= Copyright 2026 Northwestern University, 
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
using LegatePreferences: LegatePreferences

const MIN_LEGATE_VERSION = v"25.10.00"
const MAX_LEGATE_VERSION = v"25.11.00"

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

function get_version(version_file)
    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end - 2])[end])
        minor = parse(Int, lpad(split(data[end - 1])[end], 2, '0'))
        patch = parse(Int, lpad(split(data[end])[end], 2, '0'))
        version = VersionNumber(major, minor, patch)
    end
    if isnothing(version)
        error("Legate.jl: Failed to parse version for $(version_file)")
    end
    return version
end

function get_legate_version(legate_root::String)
    version_file = joinpath(legate_root, "include", "legate/legate", "version.h")
    return get_version(version_file)
end

function is_supported_version(version::VersionNumber)
    return MIN_LEGATE_VERSION <= version && version <= MAX_LEGATE_VERSION
end

function legate_valid(legate_root::String)
    # todo check if legate_root matches the version that we are installing.
    version_legate = get_legate_version(legate_root)
    return is_supported_version(version_legate)
end

# patch legion. The readme below talks about our compilation error
# https://github.com/ejmeitz/cuNumeric.jl/blob/main/scripts/README.md
function patch_legion(repo_root::String, legate_root::String)
    @info "Legate.jl: Patching Legion"

    legion_patch = joinpath(repo_root, "scripts/patch_legion.sh")
    @info "Legate.jl: Running legion patch script: $legion_patch"
    run_sh(`bash $legion_patch $repo_root $legate_root`, "legion_patch")
end

function build_jlcxxwrap(repo_root, legate_root)
    @info "libcxxwrap: Downloading"
    build_libcxxwrap = joinpath(repo_root, "scripts/install_cxxwrap.sh")

    version_path = joinpath(DEPOT_PATH[1], "dev/libcxxwrap_julia_jll/override/LEGATE_INSTALL.txt")

    if isfile(version_path)
        version = VersionNumber(strip(read(version_path, String)))
        @info "libcxxwrap: Found Legate $version"
        if is_supported_version(version)
            @info "libcxxwrap: Found supported version built with Legate.jl: $version"
            return nothing
        else
            @info "libcxxwrap: Unsupported version found: $version. Rebuilding..."
        end
    else
        @info "libcxxwrap: No version file found. Starting build..."
    end

    @info "libcxxwrap: Running build script: $build_libcxxwrap"
    run_sh(`bash $build_libcxxwrap $repo_root`, "libcxxwrap")
    open(version_path, "w") do io
        write(io, get_legate_version(legate_root))
    end
end

function build_cpp_wrapper(repo_root, legate_root, install_root)
    @info "liblegatewrapper: Building C++ Wrapper Library"
    if isdir(install_root)
        rm(install_root; recursive=true)
        mkdir(install_root)
    end

    build_cpp_wrapper = joinpath(repo_root, "scripts/build_cpp_wrapper.sh")
    nthreads = Threads.nthreads()

    bld_command = `$build_cpp_wrapper $repo_root $legate_root $install_root $nthreads`

    # write out a bash script for debugging
    cmd_str = join(bld_command.exec, " ")
    wrapper_path = joinpath(repo_root, "build_wrapper.sh")
    open(wrapper_path, "w") do io
        println(io, "#!/bin/bash")
        println(io, "set -xe")
        println(io, cmd_str)
    end
    chmod(wrapper_path, 0o755)

    @info "Running build command: $bld_command"
    run_sh(`bash $bld_command`, "cpp_wrapper")
end

function replace_nothing_jll(lib, jll)
    if isnothing(lib)
        eval(:(using $(jll)))
        jll_mod = getfield(Main, jll)
        lib = joinpath(jll_mod.artifact_dir, "lib")
    end
    return lib
end

function replace_nothing_conda_jll(mode, root, jll)
    if isnothing(root)
        if mode == LegatePreferences.MODE_CONDA
            root = load_preference(LegatePreferences, "legate_conda_env", nothing)
        else
            eval(:(using $(jll)))
            jll_mod = getfield(Main, jll)
            root = jll_mod.artifact_dir
        end
    end
    return root
end

function build(mode)
    if mode == LegatePreferences.MODE_JLL
        @warn "No reason to Build on JLL mode. Exiting Build"
        return nothing
    end

    pkg_root = up_dir(@__DIR__)
    deps_dir = joinpath(@__DIR__)

    build_log = joinpath(deps_dir, "build.log")
    open(build_log, "w") do io
        println(io, "=== Build started ===")
    end

    @info "Legate.jl: Parsed Package Dir as: $(pkg_root)"
    # can be nothing so this errors if not set
    legate_root = load_preference(LegatePreferences, "legate_path", nothing)
    legate_root = replace_nothing_conda_jll(mode, legate_root, :legate_jll)
    legate_lib = joinpath(legate_root, "lib")

    # only patch if not legate_jll
    if mode == LegatePreferences.MODE_DEVELOPER || mode == LegatePreferences.MODE_CONDA
        patch_legion(pkg_root, up_dir(legate_lib))
    end

    if mode == LegatePreferences.MODE_DEVELOPER
        install_dir = joinpath(pkg_root, "lib", "legate_jl_wrapper", "build")
        build_jlcxxwrap(pkg_root, legate_root) # $pkg_root/lib/libcxxwrap-julia 
        if !legate_valid(legate_root)
            error(
                "Legate.jl: Unsupported Legate version at $(legate_root). " *
                "Installed version: $(installed_version) not in range supported: " *
                "$(MIN_LEGATE_VERSION)-$(MAX_LEGATE_VERSION).",
            )
        end
        build_cpp_wrapper(pkg_root, legate_root, install_dir) # $pkg_root/lib/legate_jl_wrapper
    end
end

const mode = load_preference(LegatePreferences, "legate_mode", LegatePreferences.MODE_JLL)
build(mode)
