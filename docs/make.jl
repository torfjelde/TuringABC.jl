using TuringABC
using Documenter

DocMeta.setdocmeta!(TuringABC, :DocTestSetup, :(using TuringABC); recursive=true)

makedocs(;
    modules=[TuringABC],
    authors="Tor Erlend Fjelde <tor.erlend95@gmail.com> and contributors",
    repo="https://github.com/torfjelde/TuringABC.jl/blob/{commit}{path}#{line}",
    sitename="TuringABC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://torfjelde.github.io/TuringABC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "getting-started.md",
        "api.md",
    ],
)

deploydocs(;
    repo="github.com/torfjelde/TuringABC.jl",
    devbranch="main",
)
