module Legate


using CxxWrap

lib = "liblegatewrapper.so"
@wrapmodule(() -> joinpath(@__DIR__, "../", "wrapper", "build", lib))

function init()
    @initcxx
end

end