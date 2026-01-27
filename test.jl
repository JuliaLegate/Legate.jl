using FunctionWrappers
import FunctionWrappers: FunctionWrapper
using Legate

const TaskFunType = FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}

struct JuliaTask
    fun::TaskFunType
end

const my_task_ref = Ref{JuliaTask}()
const my_init_task_ref = Ref{JuliaTask}()

# Example of wrapping your original task_test
function task_test(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32})
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

function task_init(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32})
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end
end

# Store the wrapped function
const my_task = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}(task_test)
)
const my_init_task = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}(task_init)
)

my_task_ref[] = my_task
my_init_task_ref[] = my_init_task

Base.@ccallable function julia_task_fn(
    task_ptr::Ptr{Nothing},
    inputs::Ptr{Ptr{Cvoid}},
    outputs::Ptr{Ptr{Cvoid}},
    n::Int64,
)::Cvoid
    println("Starting Julia task function")

    task_ref = unsafe_pointer_to_objref(task_ptr)::Ref{JuliaTask}  # <- note Ref
    task = task_ref[]  # unwrap

    a = unsafe_wrap(Vector{Float32}, Ptr{Float32}(unsafe_load(inputs, 1)), n; own=false)
    b = unsafe_wrap(Vector{Float32}, Ptr{Float32}(unsafe_load(inputs, 2)), n; own=false)
    c = unsafe_wrap(Vector{Float32}, Ptr{Float32}(unsafe_load(outputs, 1)), n; own=false)

    task.fun(a, b, c)

    @info "Completed Julia task function"
    return nothing
end

function register_task()
    fn_ptr = @cfunction(
        julia_task_fn,
        Cvoid,
        (Ptr{Nothing}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Int64)
    )

    Legate.register_julia_task(
        50001,
        fn_ptr,
    )

    Legate.register_julia_task(
        50002,
        fn_ptr,
    )

    println("Registered Julia tasks with C++ dispatcher")
end

function test_driver()
    rt = Legate.get_runtime()
    lib = Legate.create_library("test")
    Legate.ufi_interface_register(lib)
    println("Registered library with C++ runtime")

    task = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

    a = Legate.create_array([100], Float32)
    b = Legate.create_array([100], Float32)
    c = Legate.create_array([100], Float32)

    init_output_vars = Vector{Legate.Variable}()

    push!(init_output_vars, Legate.add_output(task, a))
    push!(init_output_vars, Legate.add_output(task, b))
    push!(init_output_vars, Legate.add_output(task, c))
    Legate.add_scalar(task, Legate.Scalar(UInt32(50001))) # init task id
    task_ptr = Base.unsafe_convert(Ptr{Nothing}, my_init_task_ref)
    Legate.add_scalar(task, Legate.Scalar(task_ptr))
    Legate.default_alignment(task, Vector{Legate.Variable}(), init_output_vars)
    Legate.submit_task(rt, task)

    input_vars = Vector{Legate.Variable}()
    output_vars = Vector{Legate.Variable}()

    push!(input_vars, Legate.add_input(task, a))
    push!(input_vars, Legate.add_input(task, b))
    push!(output_vars, Legate.add_output(task, c))

    Legate.add_scalar(task, Legate.Scalar(UInt32(50002))) # main task id
    task_ptr = Base.unsafe_convert(Ptr{Nothing}, my_task_ref)
    Legate.add_scalar(task, Legate.Scalar(task_ptr))
    Legate.default_alignment(task, input_vars, output_vars)
    Legate.submit_task(rt, task)

    println("Task executed successfully")
end

if abspath(PROGRAM_FILE) == @__FILE__
    register_task()
    test_driver()
end
