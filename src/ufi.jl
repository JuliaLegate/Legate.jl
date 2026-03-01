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
const MAX_UFI_SLOTS_VAL = 32
const SLOT_REQUEST_PTRS = Vector{Ptr{Cvoid}}(undef, MAX_UFI_SLOTS_VAL)

const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_SHUTDOWN_DONE = Ref{Bool}(false)
const UFI_INIT_LOCK = ReentrantLock()
const DISPATCH_LOCK = ReentrantLock()

const IN_POLL = Threads.Atomic{Int}(0)
const PENDING_JOBS = Threads.Atomic{Int}(0)

const UFI_ERROR = 217

struct TaskJob
    slot_id::Int
    in_args::Vector{Ptr{Cvoid}}
    out_args::Vector{Ptr{Cvoid}}
    scal_args::Vector{Ptr{Cvoid}}
    meta::UfiMetadata
end

const JOB_QUEUE = Ref{Channel{TaskJob}}()
const UFI_POLLER_TASK = Ref{Task}()
const UFI_WORKER_TASKS = Vector{Task}()

"""
    _extract_and_call(meta, in_p, out_p, scal_p, sig)

The type-stable JIT engine. Reconstructs arguments from raw pointers using the 
Signature's type parameters (including dimensions). Perfectly type-stable and 
zero-allocation in the hot path.
"""
@generated function _do_call(f, in_p_ptr::Ptr{Ptr{Cvoid}}, out_p_ptr::Ptr{Ptr{Cvoid}}, scal_p_ptr::Ptr{Ptr{Cvoid}}, dims::Tuple, ::UfiSignature{InT, OutT, ScT}) where {InT, OutT, ScT}
    exprs = []
    dim_cursor = 1
    
    # Inputs
    for (i, T) in enumerate(InT.parameters)
        E = eltype(T)
        push!(exprs, :(unsafe_wrap(Array, Ptr{$E}(unsafe_load(in_p_ptr, $i)), dims[$dim_cursor])))
        dim_cursor += 1
    end
    
    # Outputs
    for (i, T) in enumerate(OutT.parameters)
        E = eltype(T)
        push!(exprs, :(unsafe_wrap(Array, Ptr{$E}(unsafe_load(out_p_ptr, $i)), dims[$dim_cursor])))
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

function _extract_and_call(meta::UfiMetadata{F, S, D}, in_args::Vector{Ptr{Cvoid}}, out_args::Vector{Ptr{Cvoid}}, scal_args::Vector{Ptr{Cvoid}}, sig::S) where {F, S, D}
    GC.@preserve in_args out_args scal_args begin
        _do_call(meta.fun, pointer(in_args), pointer(out_args), pointer(scal_args), meta.dims, sig)
    end
end

function ufi_has_pending_work()
    # Robust synchronization using submitted/started counts
    submitted = SUBMITTED_COUNT[]
    started = Int(ccall((:legate_get_started_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    
    active_calls = Int(ccall((:legate_get_active_call_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    active_slots = Int(ccall((:legate_get_active_slot_count, Legate.WRAPPER_LIB_PATH), Cint, ()))

    return submitted > started || active_calls > 0 || active_slots > 0 || PENDING_JOBS[] > 0
end

function wait_ufi()
    while ufi_has_pending_work()
        if Threads.threadid() == 1
            ufi_poll()
        end
        yield()
        sleep(0.001)
    end
end

function ufi_poll()
    if !UFI_INITIALIZED[]; return false; end
    if Threads.atomic_cas!(IN_POLL, 0, 1) != 0; return false; end
    
    slot_id = Int(ccall((:legate_pop_pending_slot_nonblocking, Legate.WRAPPER_LIB_PATH), Cint, ()))
    if slot_id != -1
        base_ptr = SLOT_REQUEST_PTRS[slot_id + 1]
        task_id = unsafe_load(Ptr{UInt32}(base_ptr + 4))
        
        meta = lock(REGISTRY_LOCK) do
            get(GLOBAL_TASK_REGISTRY, task_id, nothing)
        end
        
        if isnothing(meta)
            println(stderr, "[UFI Error] Task ID $task_id not found in registry!")
            exit(UFI_ERROR)
        end

        # Extract pointers immediately into stable vectors
        sig_type = typeof(meta.sig)
        in_p_ptr = unsafe_load(Ptr{Ptr{Ptr{Cvoid}}}(base_ptr + 8))
        out_p_ptr = unsafe_load(Ptr{Ptr{Ptr{Cvoid}}}(base_ptr + 16))
        scal_p_ptr = unsafe_load(Ptr{Ptr{Ptr{Cvoid}}}(base_ptr + 24))

        in_args = [unsafe_load(in_p_ptr, i) for i in 1:length(sig_type.parameters[1].parameters)]
        out_args = [unsafe_load(out_p_ptr, i) for i in 1:length(sig_type.parameters[2].parameters)]
        scal_args = [unsafe_load(scal_p_ptr, i) for i in 1:length(sig_type.parameters[3].parameters)]

        Threads.atomic_add!(PENDING_JOBS, 1)
        put!(JOB_QUEUE[], TaskJob(slot_id, in_args, out_args, scal_args, meta))
        return true
    end
    IN_POLL[] = 0
    return false
end

function _ufi_worker_loop()
    _is_precompiling && return
    while !UFI_SHUTDOWN[]
        job = take!(JOB_QUEUE[])
        try
            _extract_and_call(job.meta, job.in_args, job.out_args, job.scal_args, job.meta.sig)
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

function _ufi_poller_loop()
    _is_precompiling && return
    while !UFI_SHUTDOWN[]
        if !ufi_poll()
            yield()
            # sleep(0.001)
        end
    end
    UFI_SHUTDOWN_DONE[] = true
end


function init_ufi()
    lock(UFI_INIT_LOCK) do
        UFI_INITIALIZED[] && return
        _is_precompiling() && return
        
        JOB_QUEUE[] = Channel{TaskJob}(128)
        
        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        if max_slots <= 0
            exit(UFI_ERROR)
        end

        for i in 1:max_slots
            SLOT_REQUEST_PTRS[i] = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{Cvoid}, (Cint,), Cint(i-1))
        end

        LegateInternal._initialize_async_system()
        UFI_INITIALIZED[] = true
        
        precompile(_ufi_poller_loop, ())
        precompile(_ufi_worker_loop, ())

        UFI_POLLER_TASK[] = errormonitor(@async _ufi_poller_loop())

        empty!(UFI_WORKER_TASKS)
        # Ensure at least one worker loop is running, even in single-threaded mode.
        num_workers = max(1, Threads.nthreads() - 1)
        for i in 1:num_workers
            push!(UFI_WORKER_TASKS, errormonitor(Threads.@spawn _ufi_worker_loop()))
        end
        yield()

        println(stderr, "[UFI] System Initialized (Concurrent Count-Sync Mode) with $(num_workers) workers\n")
    end

    @debug "fuck"
end

function shutdown_ufi()
    UFI_SHUTDOWN[] = true
    UFI_INITIALIZED[] = false
    if isassigned(JOB_QUEUE) && isopen(JOB_QUEUE[])
        close(JOB_QUEUE[])
    end
end