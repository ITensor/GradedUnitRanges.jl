# default behavior: any object is self-dual
dual(x) = x
nondual(r::AbstractUnitRange) = r
isdual(::AbstractUnitRange) = false

dual_type(x) = dual_type(typeof(x))
dual_type(T::Type) = T
nondual_type(x) = nondual_type(typeof(x))
nondual_type(T::Type) = T

using LabelledNumbers: LabelledStyle, IsLabelled, NotLabelled, label, labelled, unlabel

dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))
label_dual(x) = label_dual(LabelledStyle(x), x)
label_dual(::NotLabelled, x) = x
label_dual(::IsLabelled, x) = labelled(unlabel(x), dual(label(x)))

flip(a::AbstractUnitRange) = dual(label_dual(a))
flip(g::AbstractGradedUnitRange) = dual(gradedrange(label_dual.(blocklengths(g))))
