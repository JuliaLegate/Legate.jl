module Legate


using CxxWrap

lib = "liblegatewrapper.so"
@wrapmodule(() -> joinpath(@__DIR__, "../", "wrapper", "build", lib))

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
end 