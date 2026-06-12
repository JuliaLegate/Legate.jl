function resolve_custom_cuda(pkg_name::String)
    cuda_enabled = Sys.which("nvcc") !== nothing
    if !cuda_enabled
        @warn "nvcc not found on PATH — building without CUDA. " *
            "If your $(pkg_name) was built with CUDA, add nvcc to PATH and rebuild."
    end
    return cuda_enabled, nothing
end

function detect_jll_cuda_enabled(jll_mod)
    cuda_val = get(jll_mod.host_platform.tags, "cuda", nothing)
    return cuda_val !== nothing && cuda_val != "none"
end

# try/catch because CUDA_SDK_jll is a weakdep — may not be in the environment.
function try_get_cuda_sdk_jll_dir()
    try
        Core.eval(Main, :(using CUDA_SDK_jll))
        return joinpath(getfield(Main, :CUDA_SDK_jll).artifact_dir, "cuda")
    catch
        return nothing
    end
end

function build_cuda_env(cuda_enabled::Bool, cuda_root)
    # user-set env vars override auto-detection
    if haskey(ENV, "LEGATE_WRAPPER_ENABLE_CUDA")
        user_enable = uppercase(ENV["LEGATE_WRAPPER_ENABLE_CUDA"]) == "ON"
        if user_enable && !cuda_enabled
            @warn "LEGATE_WRAPPER_ENABLE_CUDA=ON overrides auto-detected cuda_enabled=false. " *
                "The wrapper will link CUDA, but the underlying legate build may not — expect link/runtime errors if mismatched."
        end
        cuda_enabled = user_enable
    end
    if haskey(ENV, "CUDA_TOOLKIT_ROOT")
        cuda_root = ENV["CUDA_TOOLKIT_ROOT"]
    end

    env = Dict{String,String}()
    !cuda_enabled && (env["LEGATE_WRAPPER_ENABLE_CUDA"] = "OFF")
    if cuda_root !== nothing
        env["CUDA_TOOLKIT_ROOT"] = cuda_root
        # Prepend the SDK's bin dir so cmake detects the JLL's nvcc
        env["PATH"] = "$(joinpath(cuda_root, "bin")):\$PATH"
    end
    return env. 
end
