module BuildTools

# Parses a C++ version header whose last three lines are VERSION_MAJOR/MINOR/PATCH defines.
function get_version(version_file::String)
    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end - 2])[end])
        minor = parse(Int, lpad(split(data[end - 1])[end], 2, '0'))
        patch = parse(Int, lpad(split(data[end])[end], 2, '0'))
        version = VersionNumber(major, minor, patch)
    end
    isnothing(version) && error("BuildTools: failed to parse version from $(version_file)")
    return version
end

function run_sh(cmd::Cmd, filename::String; log_dir::String)
    println(cmd)

    build_log = joinpath(log_dir, "build.log")
    tmp_build_log = joinpath(log_dir, "$(filename).log")
    err_log = joinpath(log_dir, "$(filename).err")

    isfile(err_log) && rm(err_log)
    isfile(tmp_build_log) && rm(tmp_build_log)

    try
        run(pipeline(cmd; stdout=tmp_build_log, stderr=err_log, append=false))
        contents = read(tmp_build_log, String)
        open(build_log, "a") do io
            println(contents)
        end
    catch
        println("stderr log generated: ", err_log, '\n')
        contents = read(err_log, String)
        if !isempty(strip(contents))
            println("---- Begin stderr log ----")
            println(contents)
            println("---- End stderr log ----")
        end
    end
end

# Core.eval(Main, ...) loads the JLL into Main's namespace so getfield can find it
# regardless of which module this function is called from.
function find_jll_artifact_dir(jll::Symbol)
    Core.eval(Main, :(using $(jll)))
    return getfield(Main, jll).artifact_dir
end

function check_cmake_version(min_version::VersionNumber)
    cmake = Sys.which("cmake")
    if cmake === nothing
        error("cmake not found on PATH. Developer builds require cmake >= $(min_version).")
    end

    out = readchomp(`$cmake --version`)
    m = match(r"cmake version (\d+\.\d+\.\d+)", out)
    if m === nothing
        @warn "Could not parse cmake version from `$cmake --version`; proceeding."
        return nothing
    end

    ver = VersionNumber(m.captures[1])
    if ver < min_version
        error(
            "cmake $(ver) found at $(cmake), but developer builds require >= $(min_version). " *
            "Install a newer cmake (e.g. `pip install --upgrade cmake`) and rebuild.",
        )
    end

    @info "Found cmake $(ver) at $(cmake)"
    return nothing
end

function write_debug_script(path::String, cmd::Cmd; env::Dict{String,String}=Dict{String,String}())
    open(path, "w") do io
        println(io, "#!/bin/bash")
        println(io, "set -xe")
        for (k, v) in env
            println(io, "export $(k)=$(v)")
        end
        println(io, join(cmd.exec, " "))
    end
    chmod(path, 0o755)
end

function start_build(pkg_name::String, deps_dir::String)
    pkg_root = abspath(joinpath(deps_dir, ".."))
    open(joinpath(deps_dir, "build.log"), "w") do io
        println(io, "=== Build started ===")
    end
    @info "$(pkg_name): Parsed Package Dir as: $(pkg_root)"
    return pkg_root
end

function run_build_script(repo_root, bld_command; cuda_root=nothing, cuda_enabled=true, log_dir)
    env = Dict{String,String}()
    !cuda_enabled && (env["NO_CUDA"] = "ON")
    cuda_root !== nothing && (env["CUDA_TOOLKIT_ROOT"] = cuda_root)
    write_debug_script(joinpath(repo_root, "build_wrapper.sh"), bld_command; env)
    bash_cmd = `bash $bld_command`
    final_cmd = isempty(env) ? bash_cmd : addenv(bash_cmd, env)
    @info "Running build command: $final_cmd"
    run_sh(final_cmd, "cpp_wrapper"; log_dir)
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

function detect_jll_cuda_enabled(jll_mod)
    cuda_val = get(jll_mod.host_platform.tags, "cuda", nothing)
    return cuda_val !== nothing && cuda_val != "none"
end

function resolve_custom_cuda(pkg_name::String)
    cuda_enabled = Sys.which("nvcc") !== nothing
    if !cuda_enabled
        @warn "nvcc not found on PATH — building without CUDA. " *
            "If your $(pkg_name) was built with CUDA, add nvcc to PATH and rebuild."
    end
    return cuda_enabled, nothing
end

function resolve_jll_cuda(jll_mod)
    cuda_enabled = detect_jll_cuda_enabled(jll_mod)
    cuda_root = try_get_cuda_sdk_jll_dir()
    if isnothing(cuda_root) && cuda_enabled
        @warn "CUDA_SDK_jll not found — cmake will search for system CUDA, which may " *
            "not match the JLL. Add CUDA_SDK_jll to your environment for a reproducible build."
    end
    return cuda_enabled, cuda_root
end

# is_compatible: called with the cached VersionNumber; return true to skip rebuild.
function build_jlcxxwrap(
    repo_root::String, package_version::VersionNumber;
    log_dir::String, is_compatible::Function=(v -> true),
)
    build_libcxxwrap = joinpath(repo_root, "scripts/install_cxxwrap.sh")
    override_dir = joinpath(DEPOT_PATH[1], "dev/libcxxwrap_julia_jll/override")
    version_path = joinpath(override_dir, "LEGATE_INSTALL.txt")
    lib_path = joinpath(override_dir, "lib/libcxxwrap_julia.so")

    if isfile(lib_path)
        if isfile(version_path)
            cached = VersionNumber(strip(read(version_path, String)))
            if is_compatible(cached)
                @info "libcxxwrap: Up to date (version $cached)"
                return nothing
            end
            @info "libcxxwrap: Stale (cached=$cached). Rebuilding..."
        else
            @info "libcxxwrap: No version file found. Starting build..."
        end
    else
        @info "libcxxwrap: No libcxxwrap_julia.so found. Starting build..."
    end

    @info "libcxxwrap: Running build script: $build_libcxxwrap"
    run_sh(`bash $build_libcxxwrap $repo_root`, "libcxxwrap"; log_dir)
    mkpath(dirname(version_path))
    open(version_path, "w") do io
        write(io, string(package_version))
    end
    return nothing
end

end # module BuildTools
