module GradedUnitRanges

using BlockArrays:
  BlockArrays,
  AbstractBlockVector,
  AbstractBlockedUnitRange,
  Block,
  BlockIndex,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
  BlockedVector,
  blockedrange,
  BlockIndexRange,
  block,
  blocks,
  blockaxes,
  blockfirsts,
  blocklasts,
  blockisequal,
  blocklength,
  blocklengths,
  blockindex,
  findblock,
  findblockindex,
  combine_blockaxes,
  mortar,
  sortedunion
using Compat: allequal
using FillArrays: Fill
using LabelledNumbers:
  LabelledNumbers,
  LabelledInteger,
  LabelledUnitRange,
  LabelledStyle,
  IsLabelled,
  NotLabelled,
  label,
  label_type,
  labelled,
  labelled_isequal,
  unlabel,
  islabelled
using SplitApplyCombine: groupcount

include("blockedunitrange.jl")
include("gradedunitrange.jl")
include("dual.jl")
include("labelledunitrangedual.jl")
include("gradedunitrangedual.jl")
include("onetoone.jl")
include("fusion.jl")

end
