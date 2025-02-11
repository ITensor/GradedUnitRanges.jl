using Test: @test, @test_throws, @testset

using BlockArrays: Block, blocklength, blocklengths, blockisequal, blocks

using GradedUnitRanges:
  SectorUnitRange,
  blocklabels,
  dual,
  flip,
  isdual,
  multiplicity_range,
  nondual_sector,
  sector_type,
  sectorunitrange,
  space_isequal
using SymmetrySectors: AbstractSector, SU, quantum_dimension

Base.length(s::AbstractSector) = quantum_dimension(s)

@testset "SectorUnitRange" begin
  sr = sectorunitrange(SU((1, 0)), 2:3, false, 3)
  @test sr isa SectorUnitRange
  @test space_isequal(sr, sr)
  @test !space_isequal(sr, sectorunitrange(SU((1, 1)), 2:3, false, 3))
  @test !space_isequal(sr, sectorunitrange(SU((1, 0)), 3:3, false, 3))
  @test !space_isequal(sr, sectorunitrange(SU((1, 0)), 2:3, true, 3))
  @test !space_isequal(sr, sectorunitrange(SU((1, 0)), 2:3, false, 2))

  # accessors
  @test nondual_sector(sr) == SU((1, 0))
  @test multiplicity_range(sr) == 2:3
  @test first(sr) == 3
  @test !isdual(sr)

  @test space_isequal(
    sectorunitrange(SU((1, 0)), 2:3, false, 1), sectorunitrange(SU((1, 0)), 2:3)
  )

  # Base interface
  @test length(sr) == 6
  @test firstindex(sr) == 1
  @test lastindex(sr) == 6
  @test last(sr) == 8
  @test eltype(sr) === Int
  @test eachindex(sr) == Base.oneto(6)

  sr2 = copy(sr)
  @test sr2 isa SectorUnitRange
  @test space_isequal(sr, sr2)
  sr3 = deepcopy(sr)
  @test sr3 isa SectorUnitRange
  @test space_isequal(sr, sr3)

  # BlockArrays interface
  @test blocklength(sr) == 1
  @test blocklengths(sr) == [6]
  @test only(blocks(sr)) === sr
  @test blockisequal(sr, sr)

  # GradedUnitRanges interface
  @test sector_type(sr) === SU{3,2}
  @test sector_type(typeof(sr)) === SU{3,2}
  @test blocklabels(sr) == [SU((1, 0))]

  srd = dual(sr)
  @test nondual_sector(srd) == SU((1, 0))
  @test blocklabels(srd) == [SU((1, 1))]
  @test multiplicity_range(srd) == 2:3
  @test first(srd) == 3
  @test isdual(srd)

  srf = flip(sr)
  @test nondual_sector(srf) == SU((1, 1))
  @test blocklabels(srf) == [SU((1, 0))]
  @test multiplicity_range(srf) == 2:3
  @test first(srf) == 3
  @test isdual(srf)

  # getindex
  @test_throws BoundsError sr[0]
  @test_throws BoundsError sr[7]
  for i in 1:6
    @test sr[i] == i + 2
  end
  @test sr[2:3] == 4:5
  @test sr[Block(1)] === sr
  @test_throws BlockBoundsError sr[Block(2)]

  sr2 = sr[(:, 1)]
  @test sr2 isa SectorUnitRange
  @test space_isequal(sr2, sectorunitrange(SU((1, 0)), 2:2, false, 3))
  sr3 = sr[(:, 1:2)]
  @test sr3 isa SectorUnitRange
  @test space_isequal(sr3, sectorunitrange(SU((1, 0)), 2:3, false, 3))
end
