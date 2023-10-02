using Polynomials4ML: StatRYlmBasis, RYlmBasis, evaluate
using ACEbase.Testing: println_slim, print_tf
using Test
using StaticArrays

rand_rnn() = begin
    x, y, z = rand(), rand(), rand()
    r = sqrt(x^2 + y^2 + z^2) 
    x = x/r; y = y/r; z = z/r   
    SVector{3}(x, y, z)
end

@info("Test serial evaluate")
#for L = 0:4
    L = 2
    local rylm
    local staticrylm
    staticrylm = StatRYlmBasis(L) # implemented up to L = 4
    rylm = RYlmBasis(L)
    #for ntest = 1:30
        local x, y, z, r
        X = rand_rnn()
        # print_tf(@test evaluate(staticrylm, X) ≈ evaluate(rylm, X))
        evaluate(staticrylm, X)
    # end
#end


@info("Test batched evaluate")
L = 2
local rylm
local staticrylm
staticrylm = StatRYlmBasis(L) # implemented up to L = 4
rylm = RYlmBasis(L)
N = 64
# for ntest = 1:30
    local x, y, z, r
    bX = [rand_rnn() for _ = 1:N]
    # print_tf(@test evaluate(staticrylm, X) ≈ evaluate(rylm, X))
    evaluate(staticrylm, bX)
# end


