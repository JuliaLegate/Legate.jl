"""Return whether CUDA.jl can access a CUDA-capable GPU."""
has_cuda_gpu(show_reason::Bool=false) = CUDACore.has_cuda_gpu(show_reason)

# Only recognize an explicit CPU override; native startup chooses automatic defaults.
function _cpu_only_config(config=get(ENV, "LEGATE_CONFIG", ""))
    args = Base.shell_split(config)
    gpus = nothing
    for (i, arg) in enumerate(args)
        if arg == "--gpus" && i < length(args)
            gpus = tryparse(Int, args[i + 1])
        elseif startswith(arg, "--gpus=")
            gpus = tryparse(Int, split(arg, '='; limit=2)[2])
        end
    end
    return gpus === 0
end

function _check_cuda_version(version)
    version >= MIN_CUDA_VERSION || error(
        "Legate's GPU JLL requires CUDA $MIN_CUDA_VERSION driver capability or newer; " *
        "the loaded driver supports CUDA $version. Upgrade the NVIDIA driver or use " *
        "a supported forward-compatibility driver. Installing a newer toolkit alone is insufficient.",
    )
    return nothing
end

_check_cuda(::Mode) = nothing
function _check_cuda(::Developer)
    return load_preference(LegatePreferences, "legate_use_jll", true) ? _check_cuda(JLL()) : nothing
end
function _check_cuda(::JLL)
    _cpu_only_config() && return nothing
    get(legate_jll.host_platform.tags, "cuda", "none") == "none" && return nothing
    has_cuda_gpu(true) || error(
        "Legate's GPU JLL found no visible CUDA GPUs. Check CUDA_VISIBLE_DEVICES or run with --gpus 0."
    )
    return _check_cuda_version(CUDACore.driver_version())
end
