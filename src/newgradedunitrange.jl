
using BlockArrays: BlockArrays

struct NewGradedUnitRange{T,BlockLasts,Sector} <: AbstractGradedUnitRange{T,BlockLasts}
  nondual_labels::Vector{Sector}
  multiplicity_range::BlockedOneTo{T,BlockLasts}  # TBD offset != 1?
  isdual::Bool

  function NewGradedUnitRange(glabels, grange, gisdual)
    return new{eltype(grange),Vector{eltype(grange)},eltype(glabels)}(
      glabels, grange, gisdual
    )
  end
end

#
# Accessors
#
nondual_labels(g::NewGradedUnitRange) = g.nondual_labels
multiplicity_range(g::NewGradedUnitRange) = g.multiplicity_range
isdual(g::NewGradedUnitRange) = g.isdual

#
# GradedUnitRanges interface
#
function GradedUnitRanges.sector_type(
  ::Type{<:NewGradedUnitRange{T,BlockLasts,Sector}}
) where {T,BlockLasts,Sector}
  return Sector
end

function blocklabels(g::NewGradedUnitRange)
  return isdual(g) ? dual.(nondual_labels(g)) : nondual_labels(g)
end

function dual(g::NewGradedUnitRange)
  return NewGradedUnitRange(nondual_labels(g), multiplicity_range(g), !isdual(g))
end

function flip(g::NewGradedUnitRange)
  return NewGradedUnitRange(dual.(nondual_labels(g)), multiplicity_range(g), !isdual(g))
end

function space_isequal(g1::NewGradedUnitRange, g2::NewGradedUnitRange)
  return nondual_labels(g1) == nondual_labels(g2) &&
         blockisequal(multiplicity_range(g1), multiplicity_range(g2)) &&
         isdual(g1) == isdual(g2)
end

#
# Base interface
#
Base.length(g::NewGradedUnitRange) = sum(length.(blocks(g)))

function Base.show(io::IO, g::NewGradedUnitRange)
  return print(io, nameof(typeof(g)), blocklabels(g), multiplicity_range(g))
end

function Base.show(io::IO, ::MIME"text/plain", g::NewGradedUnitRange)
  print(io, typeof(g))
  if isdual(g)
    print(io, " dual")
  end
  println()
  return print(io, join(repr.(blocks(g)), '\n'))
end

#
# BlockArrays interface
#
function BlockArrays.mortar(v::Vector{<:SectorUnitRange})
  glabels = nondual_sector.(v)
  grange = blockedrange(length.(multiplicity_range.(v)))
  gisdual = isdual(first(v))
  # TODO add checks
  return NewGradedUnitRange(glabels, grange, gisdual)
end

function BlockArrays.blocks(g::NewGradedUnitRange)
  return sectorunitrange.(
    nondual_labels(g), blocks(multiplicity_range(g)), isdual(g), blockfirsts(g)
  )
end

BlockArrays.blocklengths(g::NewGradedUnitRange) = length.(blocks(g))

function BlockArrays.blockfirsts(g::NewGradedUnitRange)
  return vcat([1], blocklasts(g)[begin:(end - 1)])
end

function BlockArrays.blocklasts(g::NewGradedUnitRange)
  return cumsum(length.(nondual_labels(g)) .* blocklengths(multiplicity_range(g)))
end

#
# slicing
#
Base.getindex(g::NewGradedUnitRange, b::Block{1}) = blocks(g)[Int(b)]

function Base.getindex(
  g::NewGradedUnitRange, br::BlockRange{1,R}
) where {R<:Tuple{AbstractUnitRange{Int64}}}  # TODO remove ambiguities & use more generic def
  return mortar(blocks(g)[Int.(br)])
end

# TODO replace Tuple with kronecker
function Base.getindex(g::NewGradedUnitRange, bx::Tuple{<:Block{1},<:Any})
  return blocks(g)[Int(first(bx))][(:, last(bx))]
end
