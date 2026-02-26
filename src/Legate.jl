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

using FunctionWrappers
import FunctionWrappers: FunctionWrapper
using StaticArrays

include(joinpath(@__DIR__, "../deps/version.jl"))
include("utilities/preference.jl")

const SUPPORTED_INT_TYPES = Union{Int32,Int64}
const SUPPORTED_FLOAT_TYPES = Union{Float32,Float64}
const SUPPORTED_NUMERIC_TYPES = Union{SUPPORTED_INT_TYPES,SUPPORTED_FLOAT_TYPES}
const SUPPORTED_TYPES = Union{
    Bool,
    Int8,Int16,Int32,Int64,
    UInt8,UInt16,UInt32,UInt64,
    Float16,Float32,Float64,
    ComplexF32,ComplexF64,
    String,
}

# Sets the LEGATE_LIB_PATH and WRAPPER_LIB_PATH preferences based on mode
# This will also include the relevant JLLs if necessary.
MODE = load_preference(LegatePreferences, "legate_mode", LegatePreferences.MODE_JLL)
@static if MODE == LegatePreferences.MODE_JLL
    using legate_jll, legate_jl_wrapper_jll
    find_paths(
        MODE;
        legate_jll_module=legate_jll,
        legate_jll_wrapper_module=legate_jl_wrapper_jll,
    )
elseif MODE == LegatePreferences.MODE_DEVELOPER
    use_legate_jll = load_preference(LegatePreferences, "legate_use_jll", true)
    if use_legate_jll
        using legate_jll
        find_paths(MODE; legate_jll_module=legate_jll)
    else
        find_paths(MODE)
    end
elseif MODE == LegatePreferences.MODE_CONDA
    find_paths(MODE)
else
    error(
        "Legate.jl: Unknown mode $(MODE)." *
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

module LegateInternal
    using CxxWrap
    import ..WRAPPER_LIB_PATH
    @wrapmodule(() -> WRAPPER_LIB_PATH)
    function init()
        @initcxx
    end
end

# Expose C++ types to the main Legate namespace for use in other files
# Note: TaskRequest and TaskRequestPrivate are defined in ufi.jl, not C++
using .LegateInternal: Library, Variable, Constraint, LocalTaskID, GlobalTaskID,
                       AutoTask as AutoTaskImpl, ManualTask as ManualTaskImpl, StoreTarget, Shape, Scalar as ScalarImpl, Slice,
                       PhysicalStore, PhysicalArray, LogicalStoreImpl, LogicalArrayImpl,
                       LegateType, Domain, Runtime

include("utilities/type_map.jl")
# api functions and documentation
include("api/types.jl")
include("api/runtime.jl")
include("api/data.jl")
include("api/tasks.jl")

include("ufi.jl")
include("utilities/attach.jl")

## These functions guard against a user trying
## to start multiple runtimes and also to allow
## package extensions which always try to re-load

const RUNTIME_INACTIVE = false
const RUNTIME_ACTIVE = true
const _start_lock = ReentrantLock()
const _shutdown_lock = ReentrantLock()
const _runtime_ref = Ref{Bool}(RUNTIME_INACTIVE)
const _shutdown_done = Ref{Bool}(false)

runtime_started() = _runtime_ref[] == RUNTIME_ACTIVE

function _finish_runtime()
    lock(_shutdown_lock) do
        _shutdown_done[] && return
        _shutdown_done[] = true

        if !Legate.UFI_SHUTDOWN_DONE[]
            Legate.shutdown_ufi()
        end

        try
            legate_finish()
        catch e
            @error "legate_finish() failed" exception=(e, catch_backtrace())
        end

        # Safety net: _exit bypasses GC finalizers that crash in
        # LogicalStore::~LogicalStore() after the Legion runtime is gone.
        ccall(:_exit, Cvoid, (Cint,), Cint(0))
    end
end

function _start_runtime()
    # Load libraries into global namespace for C++ symbol resolution
    Libdl.dlopen(WRAPPER_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    Libdl.dlopen(LEGATE_LIB_PATH, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)

    LegateInternal.init()
    start_legate()
    init_ufi()

    Base.atexit(_finish_runtime)
    return RUNTIME_ACTIVE
end

function ensure_runtime!()
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
    LegatePreferences.check_unchanged()

    _is_precompiling() && return nothing
    ensure_runtime!()
end
end
