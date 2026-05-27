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
using Pkg
using Preferences
using LegatePreferences

include("../src/utilities/dev_tools.jl")
include("version.jl")

# patch legion. The readme below talks about our compilation error
# https://github.com/ejmeitz/cuNumeric.jl/blob/main/scripts/README.md
function patch_legion(repo_root::String, legate_root::String)
    if !check_if_patch(legate_root)
        legion_patch = joinpath(repo_root, "scripts/patch_legion.sh")
        @info "Legate.jl: Running legion patch script: $legion_patch"
        BuildTools.run_sh(
            `bash $legion_patch $repo_root $legate_root`, "legion_patch"; log_dir=@__DIR__
        )
    end
end

function build_cpp_wrapper(
    repo_root, legate_root, install_root; cuda_root=nothing, cuda_enabled=true
)
    @info "liblegatewrapper: Building C++ Wrapper Library"
    isdir(install_root) && (rm(install_root; recursive=true); mkdir(install_root))
    bld_command = `$(joinpath(repo_root, "scripts/build_cpp_wrapper.sh")) $repo_root $legate_root $install_root $(Threads.nthreads())`
    BuildTools.run_build_script(repo_root, bld_command; cuda_root, cuda_enabled, log_dir=@__DIR__)
end

function build_deps(pkg_root, legate_root; cuda_root=nothing, cuda_enabled=true)
    BuildTools.check_cmake_version(MIN_CMAKE_VERSION)
    install_dir = joinpath(pkg_root, "lib", "legate_jl_wrapper", "build")
    if !legate_valid(legate_root)
        error(
            "Legate.jl: Unsupported Legate version at $(legate_root). " *
            "Installed version: $(get_legate_version(legate_root)) not in range supported: " *
            "$(MIN_LEGATE_VERSION)-$(MAX_LEGATE_VERSION).",
        )
    end
    BuildTools.build_jlcxxwrap(
        pkg_root, get_legate_version(legate_root);
        log_dir=@__DIR__, is_compatible=is_supported_version,
    )
    build_cpp_wrapper(pkg_root, legate_root, install_dir; cuda_root, cuda_enabled)
end

function build(::LegatePreferences.JLL)
    @warn "No reason to Build on JLL mode. Exiting Build"
    return nothing
end

function build(::LegatePreferences.Conda)
    @warn "Conda Build does not currently pass our CI. Proceed with caution."
    pkg_root = BuildTools.start_build("Legate.jl", @__DIR__)

    legate_root = load_preference(LegatePreferences, "legate_conda_env", nothing)
    if isnothing(legate_root)
        error("This shouldn't happen. legate_conda_env = nothing?")
    end

    is_legate_installed(legate_root; throw_errors=true)
    patch_legion(pkg_root, legate_root)
    build_deps(pkg_root, legate_root)
end

function build(::LegatePreferences.Developer)
    pkg_root = BuildTools.start_build("Legate.jl", @__DIR__)

    legate_root = load_preference(LegatePreferences, "legate_path", nothing)
    if isnothing(legate_root)
        legate_root = BuildTools.find_jll_artifact_dir(:legate_jll)

        switch = false
        dev_project = joinpath(pkg_root, "dev")
        # this code will activate the dev enviroment that has CUDA_SDK_jll
        # we should only activate / switch IF legate_jll has a host_platform that supports CUDA
        if isdir(dev_project) && BuildTools.detect_jll_cuda_enabled(legate_jll)
            Pkg.activate(dev_project)
            Pkg.instantiate()
            switch = true
        end

        cuda_enabled, cuda_root = BuildTools.resolve_jll_cuda(legate_jll)

        if (switch)
            Pkg.activate(pkg_root)
        end
    else
        is_legate_installed(legate_root; throw_errors=true)
        patch_legion(pkg_root, legate_root)
        cuda_enabled, cuda_root = BuildTools.resolve_custom_cuda("legate")
    end
    build_deps(pkg_root, legate_root; cuda_root, cuda_enabled)
end

const mode_str = load_preference(LegatePreferences, "legate_mode", LegatePreferences.MODE_JLL)
build(LegatePreferences.to_mode(mode_str))
