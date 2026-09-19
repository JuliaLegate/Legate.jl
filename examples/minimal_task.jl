using Legate

Legate.Experimental(true)  # tasking is experimental

function task_noop(a)
    return nothing
end

Legate.ensure_runtime!()
rt = Legate.get_runtime()
lib = Legate.create_library("test")

my_noop_task = Legate.wrap_task(task_noop, Legate.CPUBackend)
a_noop = Legate.create_array([10], Float32)

task0 = Legate.create_julia_task(rt, lib, my_noop_task)
Legate.add_output(task0, a_noop)
Legate.submit_task(rt, task0)

Legate.wait_ufi()
