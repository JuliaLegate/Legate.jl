using Legate

function task_noop(a)
    return nothing
end

Legate.debug_trace("ROOT: ensure_runtime")
Legate.ensure_runtime!()
rt = Legate.get_runtime()
Legate.debug_trace("ROOT: create_library")
lib = Legate.create_library("test")
Legate.debug_trace("ROOT: wrap_task")
my_noop_task = Legate.wrap_task(task_noop)

Legate.debug_trace("ROOT: create_array")
a_noop = Legate.create_array([10], Float32)

Legate.debug_trace("ROOT: create_julia_task")
task0 = Legate.create_julia_task(rt, lib, my_noop_task)

Legate.debug_trace("ROOT: add_output")
Legate.add_output(task0, a_noop)

Legate.debug_trace("ROOT: submit_task")
Legate.submit_task(rt, task0)

Legate.debug_trace("ROOT: wait_ufi")
Legate.wait_ufi()
Legate.debug_trace("ROOT: DONE")
