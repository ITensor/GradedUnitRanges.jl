
using BlockArrays: BlockArrays

struct SectorUnitRange{T,Sector,Range<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
  nondual_sector::Sector
  multiplicity_range::Range
  isdual::Bool
  offset::T

  function SectorUnitRange(s, r, b, offset)
    return new{eltype(r),typeof(s),typeof(r)}(s, r, b, offset)
  end
end

#
# Constructors
#
sectorunitrange(s, r, b=false, offset=1) = SectorUnitRange(s, r, b, offset)

#
# accessors
#
nondual_sector(sr::SectorUnitRange) = sr.nondual_sector
multiplicity_range(sr::SectorUnitRange) = sr.multiplicity_range
isdual(sr::SectorUnitRange) = sr.isdual
Base.first(sr::SectorUnitRange) = sr.offset

#
# Base interface
#
Base.axes(sr::SectorUnitRange) = Base.oneto(length(sr))

Base.eachindex(sr::SectorUnitRange) = Base.oneto(length(sr))

Base.lastindex(sr::SectorUnitRange) = length(sr)

function Base.length(sr::SectorUnitRange)
  return length(nondual_sector(sr)) * length(multiplicity_range(sr))
end  # TBD directly quantum_dimension(nondual_sector(sr))?

Base.iterate(sr::SectorUnitRange) = iterate(first(sr):last(sr))
Base.iterate(sr::SectorUnitRange, i::Integer) = iterate(first(sr):last(sr), i)

Base.last(sr::SectorUnitRange) = first(sr) + length(sr) - 1

# slicing
Base.getindex(sr::SectorUnitRange, i::Integer) = range(sr)[i]
function Base.getindex(sr::SectorUnitRange, r::AbstractUnitRange{T}) where {T<:Integer}
  return range(sr)[r]
end

# TODO replace (:,x) indexing with kronecker(:, x)
Base.getindex(sr::SectorUnitRange, t::Tuple{Colon,<:Integer}) = sr[(:, last(t):last(t))]
function Base.getindex(sr::SectorUnitRange, t::Tuple{Colon,<:AbstractUnitRange})
  return sectorunitrange(
    nondual_sector(sr), multiplicity_range(sr)[last(t)], isdual(sr), first(sr)
  )
end

Base.range(sr::SectorUnitRange) = first(sr):last(sr)

function Base.show(io::IO, sr::SectorUnitRange)
  print(io, nameof(typeof(sr)), " ", first(sr), " .+ ")
  if isdual(sr)
    print(io, "dual(", nondual_sector(sr), ")")
  else
    print(io, nondual_sector(sr))
  end
  return print(io, " => ", multiplicity_range(sr))
end

#
# GradedUnitRanges interface
#
function blocklabels(sr::SectorUnitRange)
  return isdual(sr) ? [dual(nondual_sector(sr))] : [nondual_sector(sr)]
end

function dual(sr::SectorUnitRange)
  return sectorunitrange(nondual_sector(sr), multiplicity_range(sr), !isdual(sr), first(sr))
end

function flip(sr::SectorUnitRange)
  return sectorunitrange(
    dual(nondual_sector(sr)), multiplicity_range(sr), !isdual(sr), first(sr)
  )
end

function map_blocklabels(f, sr::SectorUnitRange)
  return sectorunitrange(
    f(nondual_sector(sr)), multiplicity_range(sr), isdual(sr), first(sr)
  )
end

sector_type(::Type{<:SectorUnitRange{T,Sector}}) where {T,Sector} = Sector

function space_isequal(sr1::SectorUnitRange, sr2::SectorUnitRange)
  return nondual_sector(sr1) == nondual_sector(sr2) &&
         isdual(sr1) == isdual(sr2) &&
         multiplicity_range(sr1) == multiplicity_range(sr2) &&
         first(sr1) == first(sr2)
end

#
# BlockArrays interface
#
BlockArrays.blocks(sr::SectorUnitRange) = [sr]

BlockArrays.blocklength(::SectorUnitRange) = 1

BlockArrays.blocklengths(sr::SectorUnitRange) = [length(sr)]  # TBD return length(multiplicity_range(sr)) ?
