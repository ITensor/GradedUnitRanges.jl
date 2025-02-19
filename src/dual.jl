using LabelledNumbers: LabelledInteger, label, labelled, unlabel

"""
    dual(x)

Take the dual of the symmetry sector, graded unit range, etc.
By default, it just returns `x`, i.e. it assumes the object
is self-dual.
"""
dual(x) = x
isdual(x) = isdual(typeof(x))
isdual(::Type) = false

dual_type(T::Type) = T
nondual_type(T::Type) = isdual(T) ? dual_type(T) : T
nondual(r::AbstractUnitRange) = map_blocklabels(nondual, r)
isdual(R::Type{<:AbstractUnitRange}) = isdual(eltype(R))
isdual(L::Type{<:LabelledInteger}) = isdual(label_type(L))

dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))
dual(a::AbstractUnitRange) = map_blocklabels(dual, a)
flip(a::AbstractUnitRange) = map_blocklabels(flip, a)

flip_dual(x) = isdual(x) ? flip(x) : x

"""
    dag(r::AbstractUnitRange)

Same as `dual(r)`.
"""
dag(r::AbstractUnitRange) = dual(r)

"""
    dag(a::AbstractArray)

Complex conjugates `a` and takes the dual of the axes.
"""
function dag(a::AbstractArray)
  a′ = similar(a, dual.(axes(a)))
  a′ .= conj.(a)
  return a′
end
