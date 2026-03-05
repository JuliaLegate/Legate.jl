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

const REALM_MAX_DIM = 6
const UFI_VERBOSE = get(ENV, "VERBOSE", "1") != "0"
const MAX_UFI_SLOTS_VAL = 32
const PhysArrPtr = Ptr{Cvoid}
const SLOT_REQUEST_PTRS = Vector{Ptr{Cvoid}}(undef, MAX_UFI_SLOTS_VAL)

const UFI_INIT_LOCK = ReentrantLock()
const DISPATCH_LOCK = ReentrantLock()
const COMPILED_LOCK = ReentrantLock()
const COMPILED_SPECIALIZATIONS = Set{Type}()

const PENDING_JOBS = Threads.Atomic{Int}(0)

const UFI_ERROR = 217

struct TaskRequest
    is_gpu::Int32
    task_id::UInt32
    inputs_ptr::Ptr{PhysArrPtr}
    outputs_ptr::Ptr{PhysArrPtr}
    scalars_ptr::Ptr{Ptr{Cvoid}}
    ndim::Int32
    dims::NTuple{REALM_MAX_DIM, Int64}
end

struct TaskJob
    slot_id::Int
    in_args::Vector{PhysArrPtr}
    out_args::Vector{PhysArrPtr}
    scal_args::Vector{Ptr{Cvoid}}
    local_dims::Tuple
    meta::UfiMetadata
end

mutable struct UfiManager
    job_queue::Channel{TaskJob}
    poller_task::Task
    worker_tasks::Vector{Task}
    shutdown::Threads.Atomic{Bool}
    shutdown_done::Threads.Atomic{Bool}

    function UfiManager(num_workers::Int)
        mgr = new(
            Channel{TaskJob}(128),
            Task(() -> nothing), # poller_task placeholder
            Vector{Task}(), # worker_tasks
            Threads.Atomic{Bool}(false),
            Threads.Atomic{Bool}(false)
        )

        mgr.poller_task = errormonitor(@async _ufi_poller_loop(mgr))

        for _ in 1:num_workers
            push!(mgr.worker_tasks, errormonitor(Threads.@spawn _ufi_worker_loop(mgr)))
        end

        return mgr
    end
end

const UFI_MANAGER = Ref{Union{Nothing, UfiManager}}(nothing)

function ufi_initialized()
    return !isnothing(UFI_MANAGER[])
end

@generated function _do_call(f, in_p_ptr::Ptr{PhysArrPtr}, out_p_ptr::Ptr{PhysArrPtr}, scal_p_ptr::Ptr{Ptr{Cvoid}}, local_dims::Tuple, dims::Tuple, ::UfiSignature{InT, OutT, ScT}) where {InT, OutT, ScT}
    exprs = []
    dim_cursor = 1

    # Inputs
    for (i, T) in enumerate(InT.parameters)
        E = eltype(T)
        push!(exprs, :(unsafe_wrap(Array, Ptr{$E}(unsafe_load(in_p_ptr, $i)), local_dims)))
        dim_cursor += 1
    end

    # Outputs
    for (i, T) in enumerate(OutT.parameters)
        E = eltype(T)
        push!(exprs, :(unsafe_wrap(Array, Ptr{$E}(unsafe_load(out_p_ptr, $i)), local_dims)))
        dim_cursor += 1
    end

    # Scalars
    for (i, T) in enumerate(ScT.parameters)
        push!(exprs, :(unsafe_load(Ptr{$T}(unsafe_load(scal_p_ptr, $i)))))
    end

    return quote
        f($(exprs...))
    end
end

function _extract_and_call(meta::UfiMetadata{F, S, D}, in_args::Vector{PhysArrPtr}, out_args::Vector{PhysArrPtr}, scal_args::Vector{Ptr{Cvoid}}, local_dims::Tuple, sig::S) where {F, S, D}
    GC.@preserve in_args out_args scal_args begin
        _do_call(meta.fun, pointer(in_args), pointer(out_args), pointer(scal_args), local_dims, meta.dims, sig)
    end
end

function ufi_has_pending_work(drain_slots::Bool=true)
    active_calls = Int(ccall((:legate_get_active_call_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    active_slots = drain_slots ? Int(ccall((:legate_get_active_slot_count, Legate.WRAPPER_LIB_PATH), Cint, ())) : 0
    return active_calls > 0 || active_slots > 0 || PENDING_JOBS[] > 0
end

function wait_ufi(drain_slots::Bool=true)
    while ufi_has_pending_work(drain_slots)
        yield()
        sleep(0.001)
    end
end

function ufi_poll(mgr::UfiManager)
    if mgr.shutdown[]; return false; end

    slot_id = Int(ccall((:legate_pop_pending_slot_nonblocking, Legate.WRAPPER_LIB_PATH), Cint, ()))
    slot_id == -1 && return false

    base_ptr = SLOT_REQUEST_PTRS[slot_id + 1]
    req = unsafe_load(Ptr{TaskRequest}(base_ptr))
    task_id = req.task_id

    lock(REGISTRY_LOCK)
    meta = try
        get(GLOBAL_TASK_REGISTRY, task_id, nothing)
    finally
        unlock(REGISTRY_LOCK)
    end

    if isnothing(meta)
        if !mgr.shutdown[]
            println(stderr, "[UFI Error] Task ID $task_id not found in registry!")
            exit(UFI_ERROR)
        end
        return false
    end

    # Dimension Extraction: Use explicit loop to avoid closure JIT.
    nd = Int(req.ndim)
    local_dims = ntuple(i -> Int(max(0, req.dims[i])), nd)

    # Signature extraction for immediate use
    sig_type = typeof(meta.sig)

    in_p_ptr = req.inputs_ptr
    out_p_ptr = req.outputs_ptr
    scal_p_ptr = req.scalars_ptr

    in_p_ptr = req.inputs_ptr
    out_p_ptr = req.outputs_ptr
    scal_p_ptr = req.scalars_ptr

    # Extract pointers immediately into stable vectors via explicit loops (JIT-safe)
    in_len = length(sig_type.parameters[1].parameters)
    in_args = Vector{PhysArrPtr}(undef, in_len)
    for i in 1:in_len
        in_args[i] = unsafe_load(in_p_ptr, i)
    end
    
    out_len = length(sig_type.parameters[2].parameters)
    out_args = Vector{PhysArrPtr}(undef, out_len)
    for i in 1:out_len
        out_args[i] = unsafe_load(out_p_ptr, i)
    end
    
    scal_len = length(sig_type.parameters[3].parameters)
    scal_args = Vector{Ptr{Cvoid}}(undef, scal_len)
    for i in 1:scal_len
        scal_args[i] = unsafe_load(scal_p_ptr, i)
    end

    Threads.atomic_add!(PENDING_JOBS, 1)
    # UFI_VERBOSE && println(stderr, "[UFI] queuing task_id=$(task_id) sig=$(sig_type) local_dims=$(local_dims) global_dims=$(meta.dims)")
    try
        put!(mgr.job_queue, TaskJob(slot_id, in_args, out_args, scal_args, local_dims, meta))
    catch e
        Threads.atomic_sub!(PENDING_JOBS, 1) # Decrement if put! fails
        if e isa InvalidStateException && e.state == :closed
            return false
        end
        rethrow(e)
    end
    return true
end

function _ufi_poller_loop(mgr::UfiManager)
    _is_precompiling() && return
    while !mgr.shutdown[]
        if !ufi_poll(mgr)
            yield()
            # sleep(0.001)
        end
    end
    mgr.shutdown_done[] = true
end

function _ufi_worker_loop(mgr::UfiManager)
    _is_precompiling() && return
    while !mgr.shutdown[]
        job = try
            take!(mgr.job_queue)
        catch ex
            if ex isa InvalidStateException && ex.state == :closed
                return
            end
            @error "Error in UFI worker loop" exception=(ex, catch_backtrace())
            break
        end

        try
            # Use invokelatest to ensure MethodInstance visibility across threads
            Base.invokelatest(_extract_and_call, job.meta, job.in_args, job.out_args, job.scal_args, job.local_dims, job.meta.sig)
        catch e
            println(stderr, "[UFI Worker Error] Slot $(job.slot_id): $e")
            Base.display_error(stderr, e, catch_backtrace())
            exit(UFI_ERROR)
        finally
            Threads.atomic_sub!(PENDING_JOBS, 1)
            ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), Cint(job.slot_id))
        end
    end
end

function init_ufi()
    lock(UFI_INIT_LOCK) do
        ufi_initialized() && return
        _is_precompiling() && return
        
        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        if max_slots <= 0
            exit(UFI_ERROR)
        end

        for i in 1:max_slots
            SLOT_REQUEST_PTRS[i] = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{Cvoid}, (Cint,), Cint(i-1))
        end

        LegateInternal._initialize_async_system()
        if isdefined(LegateInternal, :JULIA_CUSTOM_GPU_TASK)
            JULIA_CUSTOM_GPU_TASK[] = LegateInternal.JULIA_CUSTOM_GPU_TASK
        end
        
        precompile(_ufi_poller_loop, (UfiManager,))
        precompile(_ufi_worker_loop, (UfiManager,))

        # Ensure at least one worker loop is running, even in single-threaded mode.
        num_workers = max(1, Threads.nthreads() - 1)
        UFI_MANAGER[] = UfiManager(num_workers)
        
        yield()

        println(stderr, "[UFI] System Initialized (Concurrent Count-Sync Mode) with $(num_workers) workers\n")
    end
end

# These are provided by the C++ wrapper when it is loaded.
# We define stubs here that will be overwritten or used by the extension.
const JULIA_CUSTOM_GPU_TASK = Ref{Any}(nothing)
export JULIA_CUSTOM_GPU_TASK

function shutdown_ufi(mgr::UfiManager = UFI_MANAGER[])
    isnothing(mgr) && return
    # Note: caller is responsible for draining work via wait_ufi() before shutdown.
    mgr.shutdown[] = true

    if isopen(mgr.job_queue)
        close(mgr.job_queue)
    end

    if mgr === UFI_MANAGER[]
        UFI_MANAGER[] = nothing
    end
end

function ufi_has_shutdown_done()
    mgr = UFI_MANAGER[]
    isnothing(mgr) && return true
    return mgr.shutdown_done[]
end