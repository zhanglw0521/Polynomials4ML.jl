module Polynomials4ML

# -------------- Import ObjectPools stuff ---------------
using ObjectPools: acquire!, release!, 
                   FlexArray, FlexArrayCache, TSafe, ArrayPool


# -------------- import ACEbase stuff 

import ACEbase
import ACEbase: evaluate, evaluate_d, evaluate_ed, evaluate_dd, evaluate_ed2, 
                evaluate!, evaluate_d!, evaluate_ed!, evaluate_ed2!
import ACEbase.FIO: read_dict, write_dict

function natural_indices end   # could rename this get_spec or similar ... 
function index end
function orthpolybasis end
function degree end 


# some stuff to allow bases to overload some lux functionality ... 
# how much of this should go into ACEbase? 
function _valtype end 
function lux end 
function _init_luxparams end 
function _init_luxstate end 

export natural_indices, 
       index, 
       evaluate, 
       evaluate_d, 
       evaluate_dd, 
       evaluate_ed, 
       evaluate_ed2, 
       evaluate!, 
       evaluate_ed!, 
       evaluate_ed2!, 
       orthpolybasis, 
       degree 



# some interface functions that we should maybe retire 
include("interface.jl")

include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")

include("monomials.jl")

include("trig.jl")
include("rtrig.jl")

include("sphericalharmonics/sphericalharmonics.jl")
include("atomicorbitalsradials/atomicorbitalsradials.jl")

include("sparseproduct.jl")

include("lux.jl")

include("testing.jl")

end