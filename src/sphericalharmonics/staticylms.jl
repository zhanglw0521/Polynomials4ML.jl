using DynamicPolynomials: @polyvar;
using StaticPolynomials: gradient, PolynomialSystem, evaluate_and_jacobian
using MultivariatePolynomials: polynomial, constant_term

using Polynomials4ML: SVecPoly4MLBasis

export StaticRYlm

struct StatRYlmBasis{TP <: PolynomialSystem} <: SVecPoly4MLBasis
    poly::TP
end

function StatRYlmBasis(L::Int)
    @polyvar x y z;
    a0 = sqrt(3) 
    a1 = sqrt(3/4/π)
    a2 = sqrt(15 / 4 / π)
    a2 = sqrt(5) * a1 
    a20 = sqrt(5 / 16 / π)
    
    # we can move the construction outside, but I don'tthink that is important
    # I checked that precomputing r² = x^2 + y^2 + z^2 will be slightly faster, but not to much
    # so I would prefer cleaner code. If we want to sqeeze to the final bit we can precompute r².
    RYlm4 = [constant_term(a0, x), 
    polynomial(a1 * y),
    polynomial(a1 * z),
    polynomial(a1 * x),
    # Y2
    a2 * x*y,
    a2 * y*z,
    a20 * (3 * y^2 - 1), 
    a2 * x*z, 
    a2 * (x^2 - y^2) / 2,
    # Y3 
    sqrt(35/32/π) * y * (3*x^2 - y^2), 
    sqrt(105/4/π) * x*y*z, 
    sqrt(21/32/π) * y * (5 * z^2 - (x^2 + y^2 + z^2 )), 
    sqrt(7/16/π) * z * (5 * z^2 - (x^2 + y^2 + z^2 )),  
    sqrt(21/32/π) * x * (5 * z^2 - (x^2 + y^2 + z^2 )), 
    sqrt(105/4/π) * z * (x^2 - y^2), 
    sqrt(35/32/π) * x * (x^2 - 3 * y^2), 
    # Y4
    ]
    return StatRYlmBasis(PolynomialSystem(RYlm4[1:sizeY(L)]))
end


# serial evaluation
evaluate(basis::StatRYlmBasis{TP}, x::AbstractVector) where TP <: PolynomialSystem = evaluate(basis.poly, x)

evaluate_ed(basis::StaticRYlmBasis{TP}, x::AbstractVector) where TP <: PolynomialSystem = evaluate_and_jacobian(basis.poly, x)


# batch evaluation

function evaluate(basis::StatRYlmBasis{TP}, x::AbstractVector{<:AbstractVector}) where TP <: PolynomialSystem
    N = length(x)
    # we still allocate an array for two reasons
    B = _alloc(basis, x)
    @simd ivdep for i = 1:N
        B[i] = evaluate(basis, x[i])
    end
    return B
end

# overring this so that ChainRulesCore.rrule works as expected
function evaluate_ed(basis::StatRYlmBasis{TP}, x::AbstractVector{<:AbstractVector}) where TP <: PolynomialSystem
    N = length(x)
    # we still allocate an array for two reasons
    B, dB = _alloc_ed(basis, x)
    @simd ivdep for i = 1:N
        B[i], dB[i] = evaluate(basis, x[i])
    end
    return B, dB
end
