function update_project(version::String)
    Pkg.compat("legate_jl_wrapper_jll", version)
    Pkg.compat("legate_jll", version)

    path = "Project.toml"
    project = TOML.parsefile(path)
    project["version"] = version

    open(path, "w") do io
        TOML.print(io, project)
    end
end
