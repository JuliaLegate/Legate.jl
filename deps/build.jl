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
using LegatePreferences

include("version.jl")
include("build_util.jl")

function build(::LegatePreferences.JLL)
    @warn "No reason to Build on JLL mode. Exiting Build"
    return nothing
end

function build(::LegatePreferences.Conda)
    @warn "Conda Build does not currently pass our CI. Proceed with caution."
    pkg_root = _start_build()

    legate_root = load_preference(LegatePreferences, "legate_conda_env", nothing)
    if isnothing(legate_root)
        error("This shouldn't happen. legate_conda_env = nothing?")
    end

    is_legate_installed(legate_root; throw_errors=true)
    patch_legion(pkg_root, legate_root)
    build_deps(pkg_root, legate_root)
end

const mode_str = load_preference(LegatePreferences, "legate_mode", LegatePreferences.MODE_JLL)
build(LegatePreferences.to_mode(mode_str))
