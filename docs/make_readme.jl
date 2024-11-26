using Literate: Literate
using GradedUnitRanges: GradedUnitRanges

Literate.markdown(
  joinpath(pkgdir(GradedUnitRanges), "examples", "README.jl"),
  joinpath(pkgdir(GradedUnitRanges));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
