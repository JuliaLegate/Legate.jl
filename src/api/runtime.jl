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
