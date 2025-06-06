module Legate

using CxxWrap
using legate_jll

lib = "liblegatewrapper.so"
@wrapmodule(() -> joinpath(@__DIR__, "../", "wrapper", "build", lib))

include("type.jl")

function my_on_exit()
    @info "Cleaning Up Legate"
    Legate.legate_finish()
end

function __init__()
    @initcxx

    Legate.start_legate()
    @info "Started Legate"
    Base.atexit(my_on_exit)
end

function get_jll()
    return legate_jll.artifact_dir
end

end 