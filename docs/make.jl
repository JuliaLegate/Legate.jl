using Documenter, DocumenterVitepress
using Legate
using LegatePreferences

function build_cpp_docs()
    doxyfile = joinpath(@__DIR__, "Doxyfile")
    run(`doxygen $doxyfile`)
end

# this creates src/_doxygen/html
build_cpp_docs()

makedocs(;
    sitename="Legate.jl",
    authors="Ethan Meitz and David Krasowska",
    format=MarkdownVitepress(;
        repo="github.com/JuliaLegate/Legate.jl",
        devbranch="main",
        devurl="dev",
    ),
    pages=[
        "Home" => "index.md",
        "Build Options" => "install.md",
        "Back End Details" => "usage.md",
        "Public API" => "api.md",
    ],
)

builddir=joinpath(@__DIR__, "build")
# we need to move the doxygen output into the right place for DocumenterVitepress
doxygen_src = joinpath(@__DIR__, "_doxygen", "html")
mv(doxygen_src, joinpath(builddir, "1", "CppAPI"))

DocumenterVitepress.deploydocs(;
    repo="github.com/JuliaLegate/Legate.jl",
    target=builddir,
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
