module Polynomials4ML

using ObjectPools: ArrayCache, TempArray, acquire!, release!

function degree end 

include("interface.jl")

include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")

include("monomials.jl")

include("trig.jl")
include("rtrig.jl")

include("sphericalharmonics/sphericalharmonics.jl")
include("atomicorbitalsradials/atomicorbitalsradials.jl")

include("testing.jl")

end
