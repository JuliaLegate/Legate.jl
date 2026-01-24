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

function get_install_liblegate()
    return LEGATE_LIBDIR
end

function is_legate_installed(legate_dir::String; throw_errors::Bool=false)
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
        major = parse(Int, split(data[end - 2])[end])
        minor = lpad(split(data[end - 1])[end], 2, '0')
        patch = lpad(split(data[end])[end], 2, '0')
        version = "$(major).$(minor).$(patch)"
    end

    if isnothing(version)
        error("Legate.jl: Failed to parse version")
    end
    return version
end

function check_if_patch(legate_dir)
    patch = joinpath(legate_dir, "include", "legate/legate", "patch")
    if isfile(patch)
        return true
    end
    return false
end

function check_legate_install(legate_dir)
    is_legate_installed(legate_dir; throw_errors=true)

    installed_version = parse_legate_version(legate_dir)
    if installed_version ∉ SUPPORTED_LEGATE_VERSIONS
        error("Legate.jl: $(legate_dir) detected unsupported version $(installed_version)")
    end

    patch = check_if_patch(legate_dir)
    if patch == false
        error("Legate.jl: legate does not have patch. Please run Pkg.build()")
    end

    @info "Legate.jl: Found a valid install in: $(legate_dir)"
    return true
end

function check_jll(m::Module)
    if !m.is_available()
        m_host_cuda = legate_jll.host_platform["cuda"]

        if (m_host_cuda == "none")
            error(
                "$(string(m)) installed but not available on this platform.\n $(string(legate_jll.host_platform))"
            )
        end

        v_host_cuda = VersionNumber(m_host_cuda)
        valid_cuda_version = MIN_CUDA_VERSION <= v_host_cuda <= MAX_CUDA_VERSION
        if !valid_cuda_version
            error(
                "$(string(m)) installed but not available on this platform. Host CUDA ver: $(v_host_cuda) not in range supported by $(string(m)): $(MIN_CUDA_VERSION)-$(MAX_CUDA_VERSION)."
            )
        else
            error("$(string(m)) installed but not available on this platform. Unknown reason.")
        end
    end
end

function find_paths(
    mode::String;
    legate_jll_module::Union{Module,Nothing}=nothing,
    legate_jll_wrapper_module::Union{Module,Nothing}=nothing,
)
    liblegate_path, liblegate_wrapper_path = _find_paths(
        to_mode(mode), legate_jll_module, legate_jll_wrapper_module
    )
    set_preferences!(LegatePreferences, "LEGATE_LIBDIR" => liblegate_path; force=true)
    set_preferences!(
        LegatePreferences, "LEGATE_WRAPPER_LIBDIR" => liblegate_wrapper_path; force=true
    )
end

function _find_paths(
    mode::JLL,
    legate_jll_module::Module,
    legate_jll_wrapper_module::Module,
)
    check_jll(legate_jll_module)
    check_jll(legate_jll_wrapper_module)
    legate_lib_dir = joinpath(legate_jll_module.artifact_dir, "lib")
    legate_wrapper_libdir = joinpath(legate_jll_wrapper_module.artifact_dir, "lib")
    return legate_lib_dir, legate_wrapper_libdir
end

function _find_paths(
    mode::Developer,
    legate_jll_module,
    legate_jll_wrapper_module::Nothing,
)
    legate_path = ""
    use_legate_jll = load_preference(LegatePreferences, "legate_use_jll", true)

    if use_legate_jll == false
        legate_path = load_preference(LegatePreferences, "legate_path", nothing)
        check_legate_install(legate_path)
    else
        check_jll(legate_jll_module)
        legate_path = legate_jll.artifact_dir
    end

    pkg_root = abspath(joinpath(@__DIR__, "../"))
    legate_wrapper_lib = joinpath(pkg_root, "lib", "legate_jl_wrapper", "build", "lib")

    return joinpath(legate_path, "lib"), legate_wrapper_lib
end

function _find_paths(
    mode::Conda,
    legate_jll_module::Nothing,
    legate_jll_wrapper_module::Module,
)
    @warn "mode = conda may break. We are using a subset of libraries from conda."

    conda_env = load_preference(LegatePreferences, "legate_conda_env", nothing)
    isnothing(conda_env) && error(
        "legate_conda_env preference must be set in LocalPreferences.toml when using conda mode"
    )

    check_legate_install(conda_env)
    legate_path = conda_env
    check_jll(legate_jll_wrapper_module)
    legate_wrapper_lib = joinpath(legate_jll_wrapper_module.artifact_dir, "lib")

    return joinpath(legate_path, "lib"), legate_wrapper_lib
end

const DEPS_MAP = Dict(
    "HDF5" => "libhdf5",
    "MPI" => "libmpi",
    "NCCL" => "libnccl",
    "CUDA_DRIVER" => "libcuda",
    "CUDA_RUNTIME" => "libcudart",
)
function find_dependency_paths(::Type{LegatePreferences.JLL})
    results = Dict{String,String}()

    paths_to_search = copy(legate_jll.LIBPATH_list)
    # If we have CUDA support try to find some other paths
    if isdefined(legate_jll, :NCCL_jll)
        append!(paths_to_search, legate_jll.NCCL_jll.CUDA_Runtime_jll.LIBPATH_list)
        push!(
            paths_to_search,
            joinpath(legate_jll.NCCL_jll.CUDA_Runtime_jll.CUDA_Driver_jll.artifact_dir, "lib"),
        )
    end

    for (name, lib) in DEPS_MAP
        results[name] = dirname(Libdl.find_library(lib, paths_to_search))
    end
    return results
end

find_dependency_paths(::Type{LegatePreferences.Developer}) = Dict{String,String}()
find_dependency_paths(::Type{LegatePreferences.Conda}) = Dict{String,String}()
