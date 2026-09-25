using Legate
using Test

const ITERATIONS = 1_000
const ELEMENTS = 4_096

function transform_task(input, bias, output, collect)
    collect && GC.gc(false)
    @inbounds @simd for i in eachindex(output)
        value = input[i] * 0.625f0 + bias[i] * 0.17f0 + 0.0005f0
        output[i] = value + 0.03125f0 * value * value
    end
    return nothing
end

function recurrence_task(transformed, input, bias, output, a, b, c, d, e)
    @inbounds @simd for i in eachindex(output)
        output[i] =
            a * transformed[i] + b * input[i] + c * bias[i] +
            d * transformed[i] * input[i] + e * transformed[i] * transformed[i]
    end
    return nothing
end

function submit_transform!(runtime, library, wrapped, input, bias, output, collect)
    task = Legate.create_julia_task(runtime, library, wrapped)
    inputs = [Legate.add_input(task, input), Legate.add_input(task, bias)]
    outputs = [Legate.add_output(task, output)]
    Legate.add_scalar(task, Legate.Scalar(collect))
    Legate.default_alignment(task, inputs, outputs)
    return Legate.submit_task(runtime, task)
end

function submit_recurrence!(
    runtime, library, wrapped, transformed, input, bias, output, a, b, c, d, e
)
    task = Legate.create_julia_task(runtime, library, wrapped)
    inputs = [
        Legate.add_input(task, transformed),
        Legate.add_input(task, input),
        Legate.add_input(task, bias),
    ]
    outputs = [Legate.add_output(task, output)]
    for value in (a, b, c, d, e)
        Legate.add_scalar(task, Legate.Scalar(value))
    end
    Legate.default_alignment(task, inputs, outputs)
    return Legate.submit_task(runtime, task)
end

function update_reference!(transformed, output, input, bias, a, b, c, d, e)
    @inbounds for i in eachindex(output)
        value = input[i] * 0.625f0 + bias[i] * 0.17f0 + 0.0005f0
        transformed[i] = value + 0.03125f0 * value * value
        output[i] =
            a * transformed[i] + b * input[i] + c * bias[i] +
            d * transformed[i] * input[i] + e * transformed[i] * transformed[i]
    end
    return nothing
end

function main()
    Legate.Experimental(true)
    runtime = Legate.get_runtime()
    library = Legate.create_library("cpu_tasking_stress")
    transform = Legate.wrap_task(transform_task, Legate.CPUBackend)
    recurrence = Legate.wrap_task(recurrence_task, Legate.CPUBackend)

    input_ref = Float32[sin(Float32(i) * 0.071f0) for i in 1:ELEMENTS]
    bias_ref = Float32[0.5f0 * sin(Float32(3i) * 0.019f0) - 0.25f0 for i in 1:ELEMENTS]
    output_ref = similar(input_ref)
    transformed_ref = similar(input_ref)

    input = Legate.create_array([ELEMENTS], Float32)
    output = Legate.create_array([ELEMENTS], Float32)
    bias = Legate.create_array([ELEMENTS], Float32)
    transformed = Legate.create_array([ELEMENTS], Float32)
    copyto!(input, input_ref)
    copyto!(bias, bias_ref)

    submitted_before = Legate.SUBMITTED_COUNT[]
    elapsed = @elapsed for iteration in 1:ITERATIONS
        a = 0.67f0 + Float32(iteration % 6) * 0.004f0
        b = 0.19f0 - Float32(iteration % 4) * 0.003f0
        c = -0.03f0 + Float32(iteration % 3) * 0.002f0
        d = 0.02f0 + Float32(iteration % 5) * 0.001f0
        e = -0.012f0 + Float32(iteration % 4) * 0.001f0

        submit_transform!(
            runtime, library, transform, input, bias, transformed, iteration % 8 == 0
        )
        submit_recurrence!(
            runtime, library, recurrence, transformed, input, bias, output, a, b, c, d, e
        )
        update_reference!(transformed_ref, output_ref, input_ref, bias_ref, a, b, c, d, e)
        input, output = output, input
        input_ref, output_ref = output_ref, input_ref

        if iteration % 50 == 0
            Legate.wait_ufi()
            @test Array(input) ≈ input_ref rtol = 2.0f-4 atol = 2.0f-5
        end
    end

    Legate.wait_ufi()
    result = Array(input)
    @test all(isfinite, result)
    @test result ≈ input_ref rtol = 2.0f-4 atol = 2.0f-5
    @test Legate.SUBMITTED_COUNT[] - submitted_before == 2 * ITERATIONS
    @test Legate.PENDING_JOBS[] == 0
    @info "CPU tasking stress passed" iterations=ITERATIONS tasks=2 * ITERATIONS seconds=elapsed
end

main()
