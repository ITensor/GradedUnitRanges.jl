using GradedUnitRanges: GradedUnitRanges
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  Aqua.test_all(GradedUnitRanges; ambiguities=false, piracies=false)
end
