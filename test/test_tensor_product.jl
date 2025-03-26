@eval module $(gensym())
using Test: @test, @testset

using BlockArrays: blocklength, blocklengths

using GradedUnitRanges:
  GradedUnitRanges,
  GradedOneTo,
  blocklabels,
  dual,
  fusion_product,
  flip,
  gradedrange,
  space_isequal,
  isdual
using LabelledNumbers: labelled_isequal
using TensorProducts: ⊗, OneToOne, tensor_product

struct U1
  n::Int
end
GradedUnitRanges.dual(c::U1) = U1(-c.n)
Base.isless(c1::U1, c2::U1) = c1.n < c2.n
GradedUnitRanges.fuse_labels(x::U1, y::U1) = U1(x.n + y.n)
a0 = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])

@testset "GradedUnitRanges.tensor_product" begin
  GradedUnitRanges.fuse_labels(x::String, y::String) = x * y

  a = gradedrange(["x" => 2, "y" => 3])
  b = tensor_product(a, a)
  @test b isa GradedOneTo
  @test length(b) == 25
  @test blocklength(b) == 4
  @test blocklengths(b) == [4, 6, 6, 9]
  @test labelled_isequal(b, gradedrange(["xx" => 4, "yx" => 6, "xy" => 6, "yy" => 9]))

  c = tensor_product(a, a, a)
  @test c isa GradedOneTo
  @test length(c) == 125
  @test blocklength(c) == 8
  @test blocklabels(c) == ["xxx", "yxx", "xyx", "yyx", "xxy", "yxy", "xyy", "yyy"]

  @test labelled_isequal(
    tensor_product(a0, a0),
    gradedrange([
      U1(2) => 1,
      U1(3) => 3,
      U1(2) => 1,
      U1(3) => 3,
      U1(4) => 9,
      U1(3) => 3,
      U1(2) => 1,
      U1(3) => 3,
      U1(2) => 1,
    ]),
  )
end

@testset "GradedUnitRanges.fusion_product" begin
  g0 = OneToOne()
  @test fusion_product() isa OneToOne
  @test fusion_product(g0) isa OneToOne
  @test fusion_product(g0, g0) isa OneToOne

  for a in (a0, a0[1:5])
    @test labelled_isequal(fusion_product(a), gradedrange([U1(1) => 2, U1(2) => 3]))

    @test labelled_isequal(
      fusion_product(a, a), gradedrange([U1(2) => 4, U1(3) => 12, U1(4) => 9])
    )
    @test labelled_isequal(
      fusion_product(a, OneToOne()), gradedrange([U1(1) => 2, U1(2) => 3])
    )
    @test labelled_isequal(
      fusion_product(OneToOne(), a), gradedrange([U1(1) => 2, U1(2) => 3])
    )

    d = fusion_product(a, a, a)
    @test labelled_isequal(
      d, gradedrange([U1(3) => 8, U1(4) => 36, U1(5) => 54, U1(6) => 27])
    )

    @test labelled_isequal(⊗(a), gradedrange([U1(1) => 2, U1(2) => 3]))
    @test labelled_isequal(⊗(a, a), gradedrange([U1(2) => 4, U1(3) => 12, U1(4) => 9]))
    @test labelled_isequal(
      ⊗(a, a, a), gradedrange([U1(3) => 8, U1(4) => 36, U1(5) => 54, U1(6) => 27])
    )
  end
end

@testset "dual and tensor_product" begin
  for a in (a0, a0[1:5])
    ad = dual(a)

    b = tensor_product(ad)
    @test isdual(b)
    @test space_isequal(b, ad)

    b = fusion_product(ad)
    @test b isa GradedOneTo
    @test !isdual(b)
    @test space_isequal(b, gradedrange([U1(-2) => 3, U1(-1) => 2]))

    c = fusion_product(ad, ad)
    @test c isa GradedOneTo
    @test !isdual(c)
    @test space_isequal(c, gradedrange([U1(-4) => 9, U1(-3) => 12, U1(-2) => 4]))

    d = fusion_product(ad, a)
    @test !isdual(d)
    @test space_isequal(d, gradedrange([U1(-1) => 6, U1(0) => 13, U1(1) => 6]))

    e = fusion_product(a, ad)
    @test !isdual(d)
    @test space_isequal(e, d)
  end
end
end
