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

const MIN_CUDA_VERSION = v"13.0"
const MAX_CUDA_VERSION = v"13.9.999"
const MIN_LEGATE_VERSION = v"26.01.00"
const MAX_LEGATE_VERSION = v"26.12.00"

up_dir(dir::String) = abspath(joinpath(dir, ".."))

function get_version(version_file::String)
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

# 25.11 removes the need for our legion_redop patch.
# See https://docs.nvidia.com/legate/26.01/changes/2511.html
# Will leave function just in case if we need to patch any headers in the future.
function check_if_patch(legate_root::String)
    patch = joinpath(legate_root, "include", "legate/legate", "patch")
    if isfile(patch)
        return true
    end
    return false
end

function get_legate_version(legate_root::String)
    version_file = joinpath(legate_root, "include", "legate/legate", "version.h")
    return get_version(version_file)
end

function is_supported_version(version::VersionNumber)
    return MIN_LEGATE_VERSION <= version && version <= MAX_LEGATE_VERSION
end

function legate_valid(legate_root::String)
    version_legate = get_legate_version(legate_root)
    return is_supported_version(version_legate)
end

function is_legate_installed(legate_root::String; throw_errors::Bool=false)
    include_dir = joinpath(legate_root, "include")
    if !isdir(joinpath(include_dir, "legate/legate"))
        throw_errors && @error "Legate.jl: Cannot find include/legate/legate in $(legate_root)"
        return false
    end
    return true
end
