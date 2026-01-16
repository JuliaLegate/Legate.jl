using Documenter, DocumenterVitepress
using Legate
using LegatePreferences

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

DocumenterVitepress.deploydocs(;
    repo="github.com/JuliaLegate/Legate.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
