using GradedUnitRanges: GradedUnitRanges
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  GradedUnitRanges, :DocTestSetup, :(using GradedUnitRanges); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[GradedUnitRanges],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="GradedUnitRanges.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/GradedUnitRanges.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo="github.com/ITensor/GradedUnitRanges.jl", devbranch="main", push_preview=true
)
