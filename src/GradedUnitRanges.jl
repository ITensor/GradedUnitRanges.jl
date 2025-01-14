module GradedUnitRanges

export gradedrange

using BlockArrays

## using BlockArrays:
##   BlockArrays,
##   AbstractBlockVector,
##   AbstractBlockedUnitRange,
##   Block,
##   BlockIndex,
##   BlockRange,
##   BlockSlice,
##   BlockVector,
##   BlockedOneTo,
##   BlockedUnitRange,
##   BlockedVector,
##   blockedrange,
##   BlockIndexRange,
##   blocks,
##   blockaxes,
##   blockfirsts,
##   blocklasts,
##   blockisequal,
##   blocklength,
##   blocklengths,
##   findblock,
##   findblockindex,
##   mortar

include("blockedunitrange.jl")
include("gradedunitrange.jl")
include("dual.jl")
include("labelledunitrangedual.jl")
include("gradedunitrangedual.jl")
include("onetoone.jl")
include("fusion.jl")

end
