@eval module $(gensym())
using BlockArrays:
  Block,
  BlockedOneTo,
  blockaxes,
  blockedrange,
  blockfirsts,
  blockisequal,
  blocklasts,
  blocklength,
  blocklengths,
  blocks,
  findblock,
  mortar,
  combine_blockaxes
using BlockSparseArrays: BlockSparseArray
using GradedUnitRanges:
  AbstractGradedUnitRange,
  GradedUnitRanges,
  OneToOne,
  blocklabels,
  blockmergesortperm,
  blocksortperm,
  dag,
  dual,
  flip,
  gradedrange,
  isdual,
  nondual,
  sector_type
using LabelledNumbers:
  LabelledInteger, LabelledUnitRange, label, label_type, labelled, labelled_isequal, unlabel
using Test: @test, @test_broken, @testset
using TypeParameterAccessors: unspecify_type_parameters

struct U1
  n::Int
end
struct DualU1
  nondual::U1
end
GradedUnitRanges.nondual(s::U1) = s
GradedUnitRanges.nondual(s::DualU1) = s.nondual
GradedUnitRanges.dual(s::U1) = DualU1(s)
GradedUnitRanges.dual(s::DualU1) = nondual(s)
GradedUnitRanges.flip(s::U1) = DualU1(U1(-s.n))
GradedUnitRanges.flip(s::DualU1) = U1(-nondual(s).n)
GradedUnitRanges.isdual(::Type{DualU1}) = true
GradedUnitRanges.sector_type(::Type{U1}) = U1
GradedUnitRanges.sector_type(::Type{DualU1}) = U1
GradedUnitRanges.dual_type(::Type{DualU1}) = U1
GradedUnitRanges.dual_type(::Type{U1}) = DualU1
Base.isless(c1::U1, c2::U1) = c1.n < c2.n
Base.isless(c1::DualU1, c2::DualU1) = nondual(c1) < nondual(c2)

@testset "AbstractUnitRange" begin
  a0 = OneToOne()
  @test !isdual(a0)
  @test dual(a0) isa OneToOne
  @test labelled_isequal(a0, a0)
  @test labelled_isequal(a0, dual(a0))

  a = 1:3
  ad = dual(a)
  af = flip(a)
  @test !isdual(a)
  @test !isdual(ad)
  @test !isdual(dag(a))
  @test !isdual(af)
  @test ad isa UnitRange
  @test af isa UnitRange
  @test labelled_isequal(ad, a)
  @test labelled_isequal(af, a)

  a = blockedrange([2, 3])
  ad = dual(a)
  af = flip(a)
  @test !isdual(a)
  @test !isdual(ad)
  @test ad isa BlockedOneTo
  @test af isa BlockedOneTo
  @test blockisequal(ad, a)
  @test blockisequal(af, a)
end

@testset "dual GradedUnitRange" begin
  for a in
      [gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 2, U1(1) => 3])[1:5]]
    ad = dual(a)
    @test ad isa unspecify_type_parameters(typeof(a))
    @test ad isa AbstractGradedUnitRange
    @test eltype(ad) == LabelledInteger{Int,DualU1}
    @test blocklengths(ad) isa Vector
    @test sector_type(eltype(blocklengths(ad))) == sector_type(eltype(blocklengths(a)))
    @test sector_type(a) === U1
    @test sector_type(ad) === U1

    @test labelled_isequal(dual(ad), a)
    @test labelled_isequal(nondual(ad), a)
    @test labelled_isequal(nondual(a), a)
    @test labelled_isequal(ad, ad)
    @test !labelled_isequal(a, ad)
    @test !labelled_isequal(ad, a)

    @test isdual(ad)
    @test isdual(dag(a))
    @test isdual(only(axes(ad)))
    @test !isdual(a)
    @test axes(Base.Slice(a)) isa Tuple{typeof(a)}
    @test AbstractUnitRange{Int}(ad) == 1:5
    b = combine_blockaxes(ad, ad)
    @test isdual(b)
    @test b == 1:5
    @test labelled_isequal(b, ad)

    for x in iterate(ad)
      @test x == 1
      @test label(x) == dual(U1(0))
    end
    for x in iterate(ad, labelled(3, dual(U1(1))))
      @test x == 4
      @test label(x) == dual(U1(1))
    end

    @test blockfirsts(ad) == [labelled(1, U1(0)), labelled(3, dual(U1(1)))]
    @test blocklasts(ad) == [labelled(2, U1(0)), labelled(5, dual(U1(1)))]
    @test blocklength(ad) == 2
    @test blocklengths(ad) == [2, 3]
    @test blocklabels(ad) == [dual(U1(0)), dual(U1(1))]
    @test label.(blocklengths(ad)) == [dual(U1(0)), dual(U1(1))]
    @test findblock(ad, 4) == Block(2)
    @test only(blockaxes(ad)) == Block(1):Block(2)
    @test blocks(ad) == [labelled(1:2, dual(U1(0))), labelled(3:5, dual(U1(1)))]
    @test ad[4] == 4
    @test label(ad[4]) == dual(U1(1))
    @test ad[2:4] == 2:4
    @test isdual(ad[2:4])
    @test label(ad[2:4][Block(2)]) == dual(U1(1))
    @test ad[[2, 4]] == [2, 4]
    @test label(ad[[2, 4]][2]) == dual(U1(1))
    @test ad[Block(2)] == 3:5
    @test label(ad[Block(2)]) == dual(U1(1))
    @test ad[Block(1):Block(2)][Block(2)] == 3:5
    @test label(ad[Block(1):Block(2)][Block(2)]) == dual(U1(1))
    @test ad[[Block(2), Block(1)]][Block(1)] == 3:5
    @test label(ad[[Block(2), Block(1)]][Block(1)]) == dual(U1(1))
    @test ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)] == 3:4
    @test label(ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)]) == dual(U1(1))
    @test blocksortperm(a) == [Block(1), Block(2)]
    @test blocksortperm(ad) == [Block(1), Block(2)]
    @test blocklength(blockmergesortperm(a)) == 2
    @test blocklength(blockmergesortperm(ad)) == 2
    @test blockmergesortperm(a) == [Block(1), Block(2)]
    @test blockmergesortperm(ad) == [Block(1), Block(2)]

    @test isdual(ad[Block(1)])
    @test isdual(ad[Block(1)[1:1]])
    @test ad[Block(1)] isa LabelledUnitRange
    @test ad[Block(1)[1:1]] isa LabelledUnitRange
    @test label(ad[Block(2)]) == dual(U1(1))
    @test label(ad[Block(2)[1:1]]) == dual(U1(1))

    v = ad[[Block(2)[1:1]]]
    @test v isa AbstractVector{LabelledInteger{Int64,DualU1}}
    @test length(v) == 1
    @test label(first(v)) == dual(U1(1))
    @test unlabel(first(v)) == 3
    @test isdual(v[Block(1)])
    @test isdual(axes(v, 1))
    @test blocklabels(axes(v, 1)) == [dual(U1(1))]

    v = ad[mortar([Block(2)[1:1]])]
    @test v isa AbstractVector{LabelledInteger{Int64,DualU1}}
    @test isdual(axes(v, 1))  # used in view(::BlockSparseVector, [Block(1)[1:1]])
    @test label(first(v)) == dual(U1(1))
    @test unlabel(first(v)) == 3
    @test blocklabels(axes(v, 1)) == [dual(U1(1))]

    v = ad[[Block(2)]]
    @test v isa AbstractVector{LabelledInteger{Int64,DualU1}}
    @test isdual(axes(v, 1))  # used in view(::BlockSparseVector, [Block(1)])
    @test label(first(v)) == dual(U1(1))
    @test unlabel(first(v)) == 3
    @test blocklabels(axes(v, 1)) == [dual(U1(1))]

    v = ad[mortar([[Block(2)], [Block(1)]])]
    @test v isa AbstractVector{LabelledInteger{Int64,DualU1}}
    @test isdual(axes(v, 1))
    @test label(first(v)) == dual(U1(1))
    @test unlabel(first(v)) == 3
    @test blocklabels(axes(v, 1)) == [dual(U1(1)), dual(U1(0))]
  end
end

@testset "flip" begin
  for a in
      [gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 2, U1(1) => 3])[1:5]]
    ad = dual(a)
    @test labelled_isequal(flip(a), dual(gradedrange([U1(0) => 2, U1(-1) => 3])))
    @test labelled_isequal(flip(ad), gradedrange([U1(0) => 2, U1(-1) => 3]))

    @test blocklabels(a) == [U1(0), U1(1)]
    @test blocklabels(dual(a)) == [dual(U1(0)), dual(U1(1))]
    @test blocklabels(flip(a)) == [dual(U1(0)), dual(U1(-1))]
    @test blocklabels(flip(dual(a))) == [U1(0), U1(-1)]
    @test blocklabels(dual(flip(a))) == [U1(0), U1(-1)]

    @test blocklengths(a) == [2, 3]
    @test blocklengths(ad) == [2, 3]
    @test blocklengths(flip(a)) == [2, 3]
    @test blocklengths(flip(ad)) == [2, 3]
    @test blocklengths(dual(flip(a))) == [2, 3]

    @test !isdual(a)
    @test isdual(ad)
    @test isdual(flip(a))
    @test !isdual(flip(ad))
    @test !isdual(dual(flip(a)))
  end
end

@testset "dag" begin
  elt = ComplexF64
  r = gradedrange([U1(0) => 2, U1(1) => 3])
  a = BlockSparseArray{elt}(r, dual(r))
  a[Block(1, 1)] = randn(elt, 2, 2)
  a[Block(2, 2)] = randn(elt, 3, 3)
  @test isdual.(axes(a)) == (false, true)
  ad = dag(a)
  @test Array(ad) == conj(Array(a))
  @test isdual.(axes(ad)) == (true, false)
end
end
