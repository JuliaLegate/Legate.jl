module BuildTools
using Pkg
using Pkg.Artifacts: artifact_path

include("cuda_tools.jl")

function start_build(pkg_name::String, deps_dir::String)
    pkg_root = abspath(joinpath(deps_dir, ".."))
    open(joinpath(deps_dir, "build.log"), "w") do io
        println(io, "=== Build started ===")
    end
    @info "$(pkg_name): Parsed Package Dir as: $(pkg_root)"
    return pkg_root
end

struct package
    name::String
    uuid::String
    compat::String
end

const LEGATE_JLL_DEP = package("legate_jll", "e95fb1d3-fb9e-51b5-bdb8-1a812408cac9", "")
const CUNUMERIC_JLL_DEP = package("cupynumeric_jll", "2862d674-414d-5b0b-a494-b21f8deca547", "")
const CUDA_SDK_JLL_DEP = package("CUDA_SDK_jll", "6cbf2f2e-7e60-5632-ac76-dca2274e0be0", "")

with_compat(dep::package, compat::String) = package(dep.name, dep.uuid, compat)

# write dev/Project.toml from an ordered list of package entries.
function write_dev_project(dev_dir::String, deps::Vector{package})
    mkpath(dev_dir)
    open(joinpath(dev_dir, "Project.toml"), "w") do io
        println(io, "[deps]")
        for d in deps
            println(io, "$(d.name) = \"$(d.uuid)\"")
        end
        compat_deps = filter(d -> !isempty(d.compat), deps)
        if !isempty(compat_deps)
            println(io, "\n[compat]")
            for d in compat_deps
                println(io, "$(d.name) = \"$(d.compat)\"")
            end
        end
    end
end

# Loads the primary JLL, writes dev/Project.toml, and instantiates the env.
# Returns (artifact_dir, cuda_root) — cuda_root is nothing when CUDA is disabled.
# Activates pkg_root on exit.
function setup_jll_build_env(pkg_root::String, primary::package)
    dev_dir = joinpath(pkg_root, "dev")
    artifact_dir = find_jll_artifact_dir(Symbol(primary.name))
    jll_mod = getfield(Main, Symbol(primary.name))

    cuda_enabled = detect_jll_cuda_enabled(jll_mod)
    v = pkgversion(jll_mod)
    deps = package[with_compat(primary, "$(v.major).$(v.minor)")]

    cuda_compat = nothing
    if cuda_enabled
        cuda_compat = string(VersionNumber(jll_mod.host_platform["cuda"]).major)
        push!(deps, with_compat(CUDA_SDK_JLL_DEP, cuda_compat))
    end

    write_dev_project(dev_dir, deps)
    Pkg.activate(dev_dir)
    Pkg.resolve()
    Pkg.instantiate()

    cuda_root = cuda_enabled ? try_get_cuda_sdk_jll_dir() : nothing
    Pkg.activate(pkg_root)

    return artifact_dir, cuda_root
end

# calls using X_jll on core.main and grabs path of artifact dir
function find_jll_artifact_dir(jll::Symbol)
    Core.eval(Main, :(using $(jll)))
    return getfield(Main, jll).artifact_dir
end

# Redirect a JLL package to a locally built artifact by symlinking its override/ directory.
# Julia's JLL wrapper checks for override/ at precompile time; invalidating the .ji cache
# forces a recompile so the new path takes effect on the next session.
function set_jll_artifact_override(jll::Symbol, local_artifact::String)
    jll_name = string(jll)
    jll_src = Base.find_package(jll_name)
    if isnothing(jll_src)
        @warn "$jll not found in load path — skipping override. Add it as a dependency to enable automatic path wiring."
        return nothing
    end
    jll_pkg_dir = normpath(joinpath(dirname(jll_src), ".."))
    jll_override = joinpath(jll_pkg_dir, "override")

    (islink(jll_override) || isdir(jll_override)) && rm(jll_override; recursive=true)
    symlink(local_artifact, jll_override)

    jll_cache_dir = joinpath(DEPOT_PATH[1], "compiled",
        "v$(VERSION.major).$(VERSION.minor)", jll_name)
    isdir(jll_cache_dir) && foreach(
        f -> endswith(f, ".ji") && rm(f; force=true),
        readdir(jll_cache_dir; join=true),
    )
    @info "$jll: override → $local_artifact"
end

# wrapper to log stdout / stderr
# will capture and print to screen failures with the catch block
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

function check_cmake_version(min_version::VersionNumber)
    cmake = Sys.which("cmake")
    isnothing(cmake) &&
        error("cmake not found on PATH. Developer builds require cmake >= $(min_version).")

    out = readchomp(`$cmake --version`)
    m = match(r"cmake version (\d+\.\d+\.\d+)", out)
    isnothing(m) && error(
        "Could not parse cmake version from `$cmake --version`. Developer builds require cmake >= $(min_version)."
    )

    ver = VersionNumber(m.captures[1])
    ver < min_version && error(
        "cmake $(ver) found at $(cmake), but developer builds require >= $(min_version). " *
        "Install a newer cmake (e.g. `pip install --upgrade cmake`) and rebuild.",
    )

    @info "Found cmake $(ver) at $(cmake)"
end

# constructs bash command to run based on env
function run_build_wrapper_script(
    repo_root, bld_command; cuda_root=nothing, cuda_enabled=true, log_dir
)
    env = build_cuda_env(cuda_enabled, cuda_root)

    # generates build_wrapper.sh in repo_root with env vars and the build command — this is what gets executed
    write_build_script(joinpath(repo_root, "build_wrapper.sh"), bld_command; env)

    bash_cmd = Cmd(`bash ./build_wrapper.sh`; dir=repo_root)
    @info "Running build command: $bash_cmd"
    run_sh(bash_cmd, "cpp_wrapper"; log_dir)
end

function write_build_script(path::String, cmd::Cmd; env::Dict{String,String}=Dict{String,String}())
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
