using Test: @test, @test_throws, @testset

using BlockArrays: Block, blocklength, blocklengths, blockisequal, blocks, mortar

using GradedUnitRanges:
  AbstractGradedUnitRange,
  NewGradedUnitRange,
  SectorUnitRange,
  blocklabels,
  dual,
  gradedrange,
  flip,
  isdual,
  multiplicity_range,
  nondual_sector,
  sector_type,
  sectorunitrange,
  space_isequal
using SymmetrySectors: AbstractSector, SU, quantum_dimension

Base.length(s::AbstractSector) = quantum_dimension(s)

@testset "NewGradedUnitRange" begin
  sr1 = sectorunitrange(SU((1, 0)), 1:2)
  g1 = mortar([sr1])
  @test g1 isa NewGradedUnitRange
  @test blocklabels(g1) == [SU((1, 0))]
  @test blockisequal(multiplicity_range(g1), blockedrange([2]))
  @test !isdual(g1)

  @test length(g1) == 6
  @test blocklength(g1) == 1
  @test blocklengths(g1) == [6]
  @test space_isequal(only(blocks(g1)), sr1)

  sr2 = sectorunitrange(SU((2, 1)), 3:3, false, 6)
  g2 = mortar([sr1, sr2])
  @test g2 isa NewGradedUnitRange
  @test blocklabels(g2) == [SU((1, 0)), SU((2, 1))]
  @test blockisequal(multiplicity_range(g2), blockedrange([2, 1]))
  @test !isdual(g2)
  @test space_isequal(g2, g2)
  @test !space_isequal(g1, g2)

  @test length(g2) == 14
  @test blocklength(g2) == 2
  @test blocklengths(g2) == [6, 8]
  @test all(space_isequal.(blocks(g2), [sr1, sr2]))
  @test space_isequal(g2[Block(1)], sr1)
  @test space_isequal(g2[Block(2)], sr2)

  g2b = dual(g2)
  @test g2b isa NewGradedUnitRange
  @test blocklabels(g2b) == [SU((1, 1)), SU((2, 1))]
  @test blockisequal(multiplicity_range(g2b), blockedrange([2, 1]))
  @test isdual(g2b)

  g2f = flip(g2)
  @test g2f isa NewGradedUnitRange
  @test blocklabels(g2f) == [SU((1, 0)), SU((2, 1))]
  @test blockisequal(multiplicity_range(g2f), blockedrange([2, 1]))
  @test isdual(g2f)

  @test blockfirsts(g2) == first.(blocks(g2))
  @test blocklasts(g2) == [6, 14]
  @test space_isequal(mortar(blocks(g2)), g2)

  # slicing
  @test space_isequal(g2[Block.(1:2)], g2)
  @test space_isequal(g2[Block.(1:2)], g2)

  @test space_isequal(g2[(Block(1), 1)], sectorunitrange(SU((1, 0)), 1:1, false, 1))
  @test space_isequal(g2[(Block(1), 2)], sectorunitrange(SU((1, 0)), 2:2, false, 1))
  @test space_isequal(g2[(Block(1), 1:2)], sectorunitrange(SU((1, 0)), 1:2, false, 1))
  @test_throws BoundsError g2[(Block(1), 3)]

  @test space_isequal(g2[(Block(2), 1)], sectorunitrange(SU((2, 1)), 3:3, false, 6))
  @test space_isequal(g2[(Block(2), 1:1)], sectorunitrange(SU((2, 1)), 3:3, false, 6))
  @test_throws BoundsError g2[(Block(2), 2)]
  @test_throws BoundsError g2[(Block(2), 3:3)]  # misleading?
end
