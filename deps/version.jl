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
const MIN_LEGATE_VERSION = v"26.06.00"
const MAX_LEGATE_VERSION = v"26.11.999"
const MIN_CMAKE_VERSION = v"3.26.4"

up_dir(dir::String) = abspath(joinpath(dir, ".."))

function get_legate_version(legate_root::String)
    version_file = joinpath(legate_root, "include", "legate/legate", "version.h")
    return BuildTools.get_version(version_file)
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
