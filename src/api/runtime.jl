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
    time_microseconds() -> UInt64

Measure time in microseconds.
"""
time_microseconds

"""
    time_nanoseconds() -> UInt64

Measure time in nanoseconds.
"""
time_nanoseconds
