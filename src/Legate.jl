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

module Legate

using Preferences
using LegatePreferences
import LegatePreferences: Mode, JLL, Developer, Conda, to_mode
using Libdl
using CxxWrap

include(joinpath(@__DIR__, "../deps/version.jl"))
include("preference.jl")

const SUPPORTED_INT_TYPES = Union{Int32,Int64}
const SUPPORTED_FLOAT_TYPES = Union{Float32,Float64}
const SUPPORTED_NUMERIC_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES}
const SUPPORTED_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES,Bool}

# Sets the LEGATE_LIB_PATH and WRAPPER_LIB_PATH preferences based on mode
# This will also include the relevant JLLs if necessary.
@static if LegatePreferences.MODE == "jll"
    using legate_jll, legate_jl_wrapper_jll
    find_paths(
        LegatePreferences.MODE;
        legate_jll_module=legate_jll,
        legate_jll_wrapper_module=legate_jl_wrapper_jll,
    )
elseif LegatePreferences.MODE == "developer"
    use_legate_jll = load_preference(LegatePreferences, "legate_use_jll", true)
    if use_legate_jll
        using legate_jll
        find_paths(
            LegatePreferences.MODE;
            legate_jll_module=legate_jll,
            legate_jll_wrapper_module=nothing,
        )
    else
        find_paths(LegatePreferences.MODE)
    end
elseif LegatePreferences.MODE == "conda"
    using legate_jl_wrapper_jll
    find_paths(
        LegatePreferences.MODE,
        legate_jll_module=nothing,
        legate_jll_wrapper_module=legate_jl_wrapper_jll,
    )
else
    error(
        "Legate.jl: Unknown mode $(LegatePreferences.MODE)." *
        "Must be one of 'jll', 'developer', or 'conda'.",
    )
end

const LEGATE_LIBDIR = load_preference(LegatePreferences, "LEGATE_LIBDIR", nothing)
const LEGATE_WRAPPER_LIBDIR = load_preference(LegatePreferences, "LEGATE_WRAPPER_LIBDIR", nothing)

const WRAPPER_LIB_PATH = joinpath(LEGATE_WRAPPER_LIBDIR, "liblegate_jl_wrapper.so")
const LEGATE_LIB_PATH = joinpath(LEGATE_LIBDIR, "liblegate.so")

(isnothing(LEGATE_LIBDIR) || isnothing(LEGATE_WRAPPER_LIBDIR)) && error(
    "Legate.jl: LEGATE_LIBDIR or LEGATE_WRAPPER_LIBDIR preference not set. Check LocalPreferences.toml"
)

if !isfile(WRAPPER_LIB_PATH)
    error(
        "Could not find legate wrapper library. $(WRAPPER_LIB_PATH) is not a file." *
        "Check LocalPreferences.toml. If in developer mode try Pkg.build()",
    )
end

#! DO I NEED TO DLOPEN ANYTHING HERE FIRST?
@wrapmodule(() -> WRAPPER_LIB_PATH)

include("type_map.jl")
include("ufi.jl")

# api functions and documentation
include("api/types.jl")
include("api/runtime.jl")
include("api/data.jl")
include("api/tasks.jl")

### These functions guard against a user trying
### to start multiple runtimes and also to allow
## package extensions which always try to re-load

const RUNTIME_INACTIVE = -1
const RUNTIME_ACTIVE = 0
const _runtime_ref = Ref{Int}(RUNTIME_INACTIVE)
const _start_lock = ReentrantLock()

runtime_started() = _runtime_ref[] == RUNTIME_ACTIVE

function _start_runtime()
    Libdl.dlopen(LEGATE_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    Libdl.dlopen(WRAPPER_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)

    Legate.start_legate()
    @debug "Started Legate"

    LegatePreferences.maybe_warn_prerelease()

    Base.atexit(Legate.legate_finish)
    return RUNTIME_ACTIVE
end

function ensure_runtime!()
    # fast path (no lock)
    rt = _runtime_ref[]
    (rt == RUNTIME_INACTIVE) || return rt

    lock(_start_lock)
    try
        # re-check after lock
        rt = _runtime_ref[]
        (rt == RUNTIME_INACTIVE) || return rt

        rt = _start_runtime()
        _runtime_ref[] = rt
        return rt
    finally
        unlock(_start_lock)
    end
end

_is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0

function __init__()
    # @info "Legate __init__" pid=getpid() tid=Threads.threadid() precomp=_is_precompiling()

    LegatePreferences.check_unchanged()

    @initcxx

    _is_precompiling() && return nothing

    ensure_runtime!()
    Legate.init_ufi()
end
end
