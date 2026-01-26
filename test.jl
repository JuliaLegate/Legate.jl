module LegateJuliaTasks

using Libdl
using Legate
using PackageCompiler

function task_test(a::AbstractVector{Float32},
    b::AbstractVector{Float32},
    c::AbstractVector{Float32})
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

Base.@ccallable function legate_julia_task(
    inputs::Ptr{Ptr{Cvoid}},
    outputs::Ptr{Ptr{Cvoid}},
    n::Int64,
)::Cvoid
    a_ptr = unsafe_load(inputs, 1)
    b_ptr = unsafe_load(inputs, 2)
    c_ptr = unsafe_load(outputs, 1)

    a = unsafe_wrap(Vector{Float32}, Ptr{Float32}(a_ptr), n; own=false)
    b = unsafe_wrap(Vector{Float32}, Ptr{Float32}(b_ptr), n; own=false)
    c = unsafe_wrap(Vector{Float32}, Ptr{Float32}(c_ptr), n; own=false)

    task_test(a, b, c)
end

function register_task(task_id::UInt32)
    so_path = compile_shared_library()
    # Call the wrapped C++ function directly
    Legate.register_julia_task(task_id, so_path, "legate_julia_task")
    println("Registered Julia task $task_id with C++ dispatcher")
end

function register_library()
    # Call the wrapped C++ function directly
    lib = Legate.create_library("test")
    Legate.ufi_interface_register(lib)
    println("Registered library with C++ runtime")
end

function test_driver()
    task = Legate.create_task(rt, lib, Legate.LocalTaskID(55))
    a = Legate.create_store((100), Float32)
    b = Legate.create_store((100), Float32)
    c = Legate.create_store((100), Float32)

    input_vars = Vector{Legate.Variable}()
    output_vars = Vector{Legate.Variable}()

    push!(input_vars, Legate.add_input(task, a))
    push!(input_vars, Legate.add_input(task, b))
    push!(output_vars, Legate.add_output(task, c))

    Legate.default_alignment(task, input_vars, output_vars)
    Legate.submit_task(rt, task)

    println("Task executed successfully")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .LegateJuliaTasks

    # Step 1: register the library
    LegateJuliaTasks.register_library()

    # Step 2: register a Julia task with ID 50001
    LegateJuliaTasks.register_task(UInt32(50001))

    # Step 3: run local test
    LegateJuliaTasks.test_driver()
end
