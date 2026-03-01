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

"""
    start_legate()

Start the Legate runtime. This function is called automatically 
when the module is loaded.
"""
function start_legate()
    LegateInternal.start_legate()
end

"""
    legate_finish() -> Int32

Finalize the Legate runtime.

Returns an integer status code from the runtime shutdown procedure.
"""
function legate_finish()
    LegateInternal.legate_finish()
    return nothing
end


"""
    get_runtime() -> CxxPtr{Runtime}

Returns the Legate runtime.
"""
function get_runtime()
    LegateInternal.get_runtime()
end

"""
    has_started() -> Bool

Returns true if the Legate runtime has started.
"""
function has_started()
    LegateInternal.has_started()
end

"""
    has_finished() -> Bool

Returns true if the Legate runtime has finished.
"""
function has_finished()
    LegateInternal.has_finished()
end

"""
    create_library(name::String) -> Library

Creates a library in the runtime and registers the UFI interface
with the C++ runtime.
"""
function create_library(name::String)
    # Warn if Legate is configured for concurrency but Julia is single-threaded (for ufi usage)
    legion_config = get(ENV, "LEGION_CONFIG", "")
    m = match(r"(?:-ll:cpu|--cpus)\s+(\d+)", legion_config)
    if m !== nothing
        cpu_count = parse(Int, m.captures[1])
        if cpu_count > 1 && Threads.nthreads() == 1
            @warn "[Library: $name] Legate is configured with $cpu_count CPUs via --cpus/-ll:cpu, but Julia is using only 1 thread. " *
                  "Custom tasks will execute serially. For parallelism, launch Julia with `JULIA_NUM_THREADS=$cpu_count` (or greater)."
        end
    end

    rt = get_runtime()
    lib = LegateInternal._create_library(rt, name)
    # registers JuliaCustomTask::cpu_variant to legate runtime
    LegateInternal._ufi_interface_register(lib)
    @debug "Registered library with C++ runtime"
    return lib
end

"""
    time_microseconds() -> Int64

Returns the current time in microseconds.
"""
function time_microseconds()
    LegateInternal.time_microseconds()
end

"""
    time_nanoseconds() -> Int64

Returns the current time in nanoseconds.
"""
function time_nanoseconds()
    LegateInternal.time_nanoseconds()
end

"""
    issue_execution_fence(; blocking::Bool=true)

Issues an execution fence to the runtime.

# Arguments
- `blocking`: If true, the fence will block until all tasks are completed.
"""
function issue_execution_fence(; blocking::Bool=true)
    if blocking
        @threadcall((:legate_issue_execution_fence_blocking, Legate.WRAPPER_LIB_PATH), Cvoid, ())
    else
        LegateInternal.issue_execution_fence(false)
    end
end
"""
    wait_handle(handle::Ptr{Cvoid})

Block the calling task until the Legate operation represented by the handle is complete.
This function polls the UFI system while waiting to ensure background progress.
"""
function wait_handle(handle::Ptr{Cvoid})
    handle == C_NULL && return
    while ccall((:legate_is_handle_ready, Legate.WRAPPER_LIB_PATH), Cint, (Ptr{Cvoid},), handle) == 0
        yield()
    end
end
