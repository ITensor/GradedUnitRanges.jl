using BlockArrays: AbstractBlockedUnitRange, blocklengths
using LabelledNumbers: LabelledInteger, label, labelled
using SplitApplyCombine: groupcount
using TensorProducts: TensorProducts, tensor_product

flip_dual(r::AbstractUnitRange) = r
flip_dual(r::GradedUnitRangeDual) = flip(r)

function fuse_labels(x, y)
  return error(
    "`fuse_labels` not implemented for object of type `$(typeof(x))` and `$(typeof(y))`."
  )
end

function fuse_blocklengths(x::LabelledInteger, y::LabelledInteger)
  # return blocked unit range to keep non-abelian interface
  return blockedrange([labelled(x * y, fuse_labels(label(x), label(y)))])
end

function TensorProducts.tensor_product(
  a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange
)
  nested = map(Iterators.flatten((Iterators.product(blocks(a1), blocks(a2)),))) do it
    return mapreduce(length, fuse_blocklengths, it)
  end
  new_blocklengths = mapreduce(blocklengths, vcat, nested)
  return blockedrange(new_blocklengths)
end

# convention: sort GradedUnitRangeDual according to nondual blocks
function blocksortperm(a::AbstractUnitRange)
  return Block.(sortperm(blocklabels(nondual(a))))
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function groupsortperm(v; kwargs...)
  perm = sortperm(v; kwargs...)
  v_sorted = @view v[perm]
  group_lengths = collect(groupcount(identity, v_sorted))
  return BlockVector(perm, group_lengths)
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedUnitRangesExt`.
# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function blockmergesortperm(a::AbstractUnitRange)
  return Block.(groupsortperm(blocklabels(nondual(a))))
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedUnitRangesExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function blockmergesort(g::AbstractGradedUnitRange)
  glabels = blocklabels(g)
  gblocklengths = blocklengths(g)
  new_blocklengths = map(sort(unique(glabels))) do la
    return labelled(sum(gblocklengths[findall(==(la), glabels)]; init=0), la)
  end
  return gradedrange(new_blocklengths)
end

blockmergesort(g::GradedUnitRangeDual) = flip(blockmergesort(flip(g)))
blockmergesort(g::AbstractUnitRange) = g

# fusion_product produces a sorted, non-dual GradedUnitRange
function fusion_product(g1, g2)
  return blockmergesort(tensor_product(g1, g2))
end

fusion_product(g::AbstractUnitRange) = blockmergesort(g)
fusion_product(g::GradedUnitRangeDual) = fusion_product(flip(g))

# recursive fusion_product. Simpler than reduce + fix type stability issues with reduce
function fusion_product(g1, g2, g3...)
  return fusion_product(fusion_product(g1, g2), g3...)
end
