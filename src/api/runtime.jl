"""
    start_legate()

Start the Legate runtime.

This function initializes the Legate runtime and must be called
before creating tasks or data objects.
"""
start_legate

"""
    legate_finish() -> Int32

Finalize the Legate runtime.

Returns an integer status code from the runtime shutdown procedure.
"""
legate_finish

"""
    get_runtime() -> Runtime

Return the current Legate runtime instance.

This returns a handle to the singleton `Runtime` object managed by Legate.
"""
get_runtime

"""
    has_started() -> Bool

Check whether the Legate runtime has started.
"""
has_started

"""
    has_finished() -> Bool

Check whether the Legate runtime has finished.
"""
has_finished

"""
    create_library(name::String) -> Library

Creates a library in the runtime and registers the UFI interface
with the C++ runtime.
"""
function create_library(name::String)
    rt = get_runtime()
    lib = _create_library(rt, name) # cxxwrap call
    # registers JuliaCustomTask::cpu_variant to legate runtime
    _ufi_interface_register(lib) # cxxwrap call
    request_ptr = _get_request_ptr()
    # initialize async system to handle Julia task requests
    _initialize_async_system(request_ptr) # cxxwrap call
    @debug "Registered library with C++ runtime"
    return lib
end

"""
    time_microseconds() -> UInt64

Measure time in microseconds.
"""
time_microseconds

"""
    time_nanoseconds() -> UInt64

Measure time in nanoseconds.
"""
time_nanoseconds

"""
    issue_execution_fence(; blocking::Bool = true)

Issue an execution fence.
"""
# issue_execution_fence(; blocking::Bool = true) = issue_execution_fence(blocking)

function issue_execution_fence(blocking::Bool)
    rt_ptr = get_obj_ptr(get_runtime()[])
    Base.@threadcall(:issue_execution_fence, Cvoid, (Ptr{Cvoid}, Bool), rt_ptr, blocking)
end
