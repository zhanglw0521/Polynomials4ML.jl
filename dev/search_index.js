var documenterSearchIndex = {"docs":
[{"location":"api/#Public-API","page":"API","title":"Public API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"This page documents the public API, i.e. the list of bases and functions that are considered relatively stable and for which we aim to strictly impose semver backward compatibility. The basis sets that are considered stable are the following (please see inline documentation for initialization): ","category":"page"},{"location":"api/","page":"API","title":"API","text":"Several classes of orthogonal polynomials OrthPolyBasis1D3T\nGeneral Jacobi jacobi_basis \nLegendre legendre_basis\nChebyshev chebyshev_basis \nDiscrete distribution orthpolybasis \nComplex trigonometric polynomials CTrigBasis\nReal trigonometric polynomials RTrigBasis \nComplex spherical harmonics CYlmBasis\nReal spherical harmonics RYlmBasis ","category":"page"},{"location":"api/#In-place-Evaluation","page":"API","title":"In-place Evaluation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"This section documents the in-place evaluation interface. All basis sets implemented in this package should provide this interface as a minimal requirement. ","category":"page"},{"location":"api/","page":"API","title":"API","text":"evaluate!(P, basis, X)\nevaluate_ed!(P, dP, basis, X)\nevaluate_ed2!(P, dP, ddP, basis, X)","category":"page"},{"location":"api/","page":"API","title":"API","text":"basis : an object defining one of the basis sets \nX : a single input or array of inputs. \nP : array containing the basis values \ndP : array containing derivatives of basis w.r.t. inputs \nddP : array containing second derivatives of basis w.r.t. inputs ","category":"page"},{"location":"api/","page":"API","title":"API","text":"If X is a single input then this should normally be a Number or a StaticArray to distinguish it from collections of inputs. X can also be an AbstractArray of admissible inputs, e.g., Vector{<: Number}. ","category":"page"},{"location":"api/","page":"API","title":"API","text":"If X is a single input then P, dP, ddP will be AbstractVector. If X is an AbstractVector of inputs then P, dP, ddP must be AbstractMatrix, and so forth. ","category":"page"},{"location":"api/","page":"API","title":"API","text":"The output arrays P, dP, ddP must be sufficiently large in each dimension to accomodate the size of the input and the size of the basis, but the sizes need not match exactly. It is up to the caller to ensure matching array sizes if this is needed.","category":"page"},{"location":"api/#Allocating-Evaluation","page":"API","title":"Allocating Evaluation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"This section documents the allocating evaluation interface. All basis sets should implement this interface.","category":"page"},{"location":"api/","page":"API","title":"API","text":"P = evaluate(basis, X)\nP, dP = evaluate_ed(basis, X)\nP, dP, ddP = evaluate_ed2(basis, X)","category":"page"},{"location":"api/","page":"API","title":"API","text":"The output types of P, dP, ddP are guarnateed to be AbstractArrays but may otherwise change between package versions. The exact type should not be relied upon when using this package. ","category":"page"},{"location":"api/","page":"API","title":"API","text":"The meaning of the different symbols is exactly the same as described above. The only difference is that the output containers P, dP, ddP are now allocated.  Their type should be stable (if not, please file a bug report), but unspecified in the sense that the output type is not semver-stable for the time being.  If you need a sem-ver stable output then it is best to follow the above with a collect.","category":"page"},{"location":"backup/","page":"-","title":"-","text":"Polynomials4ML.OrthPolyBasis1D3T\nPolynomials4ML.chebyshev_basis\nPolynomials4ML.legendre_basis\nPolynomials4ML.jacobi_basis\nPolynomials4ML.MonoBasis\nPolynomials4ML.CTrigBasis\nPolynomials4ML.RTrigBasis\nPolynomials4ML.CYlmBasis\nPolynomials4ML.RYlmBasis","category":"page"},{"location":"backup/#Polynomials4ML.OrthPolyBasis1D3T","page":"-","title":"Polynomials4ML.OrthPolyBasis1D3T","text":"OrthPolyBasis1D3T: defines a basis of polynomials in terms of a 3-term recursion, \n\nbeginaligned\n   P_1(x) = A_1  \n   P_2 = A_2 x + B_2 \n   P_n = (A_n x + B_n) P_n-1(x) + C_n P_n-2(x)\nendaligned\n\nTypically (but not necessarily) such bases are obtained by orthogonalizing the monomials with respect to a user-specified distribution, which can be either continuous or discrete but must have a density function. See also \n\nlegendre_basis\nchebyshev_basis\njacobi_basis\n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML.chebyshev_basis","page":"-","title":"Polynomials4ML.chebyshev_basis","text":"chebyshev_basis(N::Integer): \n\nConstructs an OrthPolyBasis1D3T object representing a possibly rescaled version of the basis of Chebyshev polynomials of the first kind. N is the length of the basis, not the degree. \n\nCareful: the normalisation may be non-standard. \n\n\n\n\n\n","category":"function"},{"location":"backup/#Polynomials4ML.legendre_basis","page":"-","title":"Polynomials4ML.legendre_basis","text":"legendre_basis(N::Integer): \n\nConstructs an OrthPolyBasis1D3T object representing a possibly rescaled version of the basis of Legendre polynomials (L2 orthonormal on [-1, 1]). N is the length of the basis, not the degree. \n\nCareful: the normalisation may be non-standard. \n\n\n\n\n\n","category":"function"},{"location":"backup/#Polynomials4ML.jacobi_basis","page":"-","title":"Polynomials4ML.jacobi_basis","text":"jacobi_basis(N::Integer, α::Real, β::Real): \n\nConstructs an OrthPolyBasis1D3T object representing a possibly rescaled version of the basis of Jacobi polynomials Jαβ. N is the length of the basis, not the degree. \n\nCareful: the normalisation may be non-standard. \n\n\n\n\n\n","category":"function"},{"location":"backup/#Polynomials4ML.MonoBasis","page":"-","title":"Polynomials4ML.MonoBasis","text":"Standard Monomials basis. This should very rarely be used. Possibly useful in combination with a transformation of the inputs, e.g. exponential.\n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML.CTrigBasis","page":"-","title":"Polynomials4ML.CTrigBasis","text":"Complex trigonometric polynomials up to degree N (inclusive). The basis is  constructed in the order \n\n[1, exp(im*θ), exp(-im*θ), exp(2im*θ), exp(-2im*θ), ..., \n                                exp(N*im*θ), exp(-N*im*θ) ]\n\nwhere θ is input variable. \n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML.RTrigBasis","page":"-","title":"Polynomials4ML.RTrigBasis","text":"RTrigBasis(N): \n\nReal trigonometric polynomials up to degree N (inclusive). The basis is ordered as \n\n[1, cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(Nθ), sin(Nθ) ]\n\nwhere θ is input variable. \n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML.CYlmBasis","page":"-","title":"Polynomials4ML.CYlmBasis","text":"CYlmBasis(maxL, T=Float64):\n\nComplex spherical harmonics; see tests to see how they are normalized, and  idx2lm on how they are ordered. The ordering is not guarenteed to be semver-stable.\n\nThe input variable is normally an rr::SVector{3, T}. This rr need not be normalized (i.e. on the unit sphere). The derivatives account for this, i.e. they are valid even when norm(rr) != 1.\n\nmaxL : maximum degree of the spherical harmonics\nT : type used to store the coefficients for the associated legendre functions\n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML.RYlmBasis","page":"-","title":"Polynomials4ML.RYlmBasis","text":"RYlmBasis(maxL, T=Float64):\n\nReal spherical harmonics; see tests to see how they are normalized, and  idx2lm on how they are ordered. The ordering is not guarenteed to be semver-stable.\n\nThe input variable is normally an rr::SVector{3, T}. This rr need not be normalized (i.e. on the unit sphere). The derivatives account for this, i.e. they are valid even when norm(rr) != 1.\n\nmaxL : maximum degree of the spherical harmonics\nT : type used to store the coefficients for the associated legendre functions\n\n\n\n\n\n","category":"type"},{"location":"backup/","page":"-","title":"-","text":"Modules = [Polynomials4ML]","category":"page"},{"location":"backup/#Polynomials4ML.ALPolynomials","page":"-","title":"Polynomials4ML.ALPolynomials","text":"ALPolynomials : an auxiliary datastructure for evaluating the associated Legendre functions used for the spherical and solid harmonics. Constructor:\n\nALPolynomials(maxL::Integer, T::Type=Float64)\n\nThis is not part of the public API and not guaranteed to be semver-stable. Only the resulting harmonics that use the ALPs are guaranteed to be backward  compatible. \n\nImportant Note: evaluate_ed!` does NOT return derivatives, but rather  produces rescaled derivatives for better numerical stability near the poles.  See comments in code for details on how to use the ALP derivatives correctly. \n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML.SphericalCoords","page":"-","title":"Polynomials4ML.SphericalCoords","text":"struct SphericalCoords : a simple datatype storing spherical coordinates of a point (x,y,z) in the format (r, cosφ, sinφ, cosθ, sinθ). Use spher2cart and cart2spher to convert between cartesian and spherical coordinates.\n\n\n\n\n\n","category":"type"},{"location":"backup/#Polynomials4ML._init_luxparams-Tuple{Random.AbstractRNG, Any}","page":"-","title":"Polynomials4ML._init_luxparams","text":"a fall-back method for initalparameters that all AbstractPoly4MLBasis should overload \n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.cYlm!-Tuple{Any, Any, Polynomials4ML.SphericalCoords, AbstractVector, CYlmBasis}","page":"-","title":"Polynomials4ML.cYlm!","text":"evaluate complex spherical harmonics\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.cYlm!-Union{Tuple{T}, Tuple{Any, Any, AbstractArray{Polynomials4ML.SphericalCoords{T}, 1}, AbstractMatrix, Any}} where T","page":"-","title":"Polynomials4ML.cYlm!","text":"evaluate complex spherical harmonics\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.cYlm_ed!-Tuple{Any, Any, Any, Polynomials4ML.SphericalCoords, Any, Any, CYlmBasis}","page":"-","title":"Polynomials4ML.cYlm_ed!","text":"evaluate gradients of complex spherical harmonics\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.cYlm_ed!-Union{Tuple{T}, Tuple{Any, Any, Any, AbstractArray{Polynomials4ML.SphericalCoords{T}, 1}, AbstractMatrix, AbstractMatrix, CYlmBasis}} where T","page":"-","title":"Polynomials4ML.cYlm_ed!","text":"evaluate gradients of complex spherical harmonics\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.chebyshev_basis-Tuple{Integer}","page":"-","title":"Polynomials4ML.chebyshev_basis","text":"chebyshev_basis(N::Integer): \n\nConstructs an OrthPolyBasis1D3T object representing a possibly rescaled version of the basis of Chebyshev polynomials of the first kind. N is the length of the basis, not the degree. \n\nCareful: the normalisation may be non-standard. \n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.dspher_to_dcart-Tuple{Any, Any, Any}","page":"-","title":"Polynomials4ML.dspher_to_dcart","text":"convert a gradient with respect to spherical coordinates to a gradient with respect to cartesian coordinates\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.idx2l-Tuple{Integer}","page":"-","title":"Polynomials4ML.idx2l","text":"Partial inverse of lm2idx: given an index into a vector of Ylm values, return the  l index. \n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.idx2lm-Tuple{Integer}","page":"-","title":"Polynomials4ML.idx2lm","text":"Inverse of lm2idx: given an index into a vector of Ylm values, return the  l, m indices.\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.index_p-Tuple{Integer, Integer}","page":"-","title":"Polynomials4ML.index_p","text":"index_p(l,m): Return the index into a flat array of Associated Legendre Polynomials P_l^m for the given indices (l,m). P_l^m are stored in l-major order i.e. \n\n\t[P(0,0), [P(1,0), P(1,1), P(2,0), ...]\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.jacobi_basis-Tuple{Integer, Real, Real}","page":"-","title":"Polynomials4ML.jacobi_basis","text":"jacobi_basis(N::Integer, α::Real, β::Real): \n\nConstructs an OrthPolyBasis1D3T object representing a possibly rescaled version of the basis of Jacobi polynomials Jαβ. N is the length of the basis, not the degree. \n\nCareful: the normalisation may be non-standard. \n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.legendre_basis-Tuple{Integer}","page":"-","title":"Polynomials4ML.legendre_basis","text":"legendre_basis(N::Integer): \n\nConstructs an OrthPolyBasis1D3T object representing a possibly rescaled version of the basis of Legendre polynomials (L2 orthonormal on [-1, 1]). N is the length of the basis, not the degree. \n\nCareful: the normalisation may be non-standard. \n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.lm2idx-Tuple{Integer, Integer}","page":"-","title":"Polynomials4ML.lm2idx","text":"lm2idx(l,m): Return the index into a flat array of real spherical harmonics Y_lm for the given indices (l,m). Y_lm are stored in l-major order i.e.\n\n\t[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.lux-Tuple{Polynomials4ML.AbstractPoly4MLBasis}","page":"-","title":"Polynomials4ML.lux","text":"lux(basis) : convert a basis / embedding object into a lux layer. This assumes  that the basis accepts a number or short vector as input and produces an output  that is a vector. It also assumes that batched operations are implemented,  as well as some other functionality. \n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.maxL-Tuple{Union{CYlmBasis, RYlmBasis}}","page":"-","title":"Polynomials4ML.maxL","text":"max L degree for which the alp coefficients have been precomputed\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.rYlm!-Tuple{Any, Any, Any, AbstractVector, RYlmBasis}","page":"-","title":"Polynomials4ML.rYlm!","text":"evaluate real spherical harmonics\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.rYlm_ed!-Tuple{Any, Any, Any, Polynomials4ML.SphericalCoords, Any, Any, RYlmBasis}","page":"-","title":"Polynomials4ML.rYlm_ed!","text":"evaluate gradients of real spherical harmonics\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.sizeP-Tuple{Any}","page":"-","title":"Polynomials4ML.sizeP","text":"sizeP(maxL):  Return the size of the set of Associated Legendre Polynomials P_l^m(x) of degree less than or equal to the given maximum degree\n\n\n\n\n\n","category":"method"},{"location":"backup/#Polynomials4ML.sizeY-Tuple{Any}","page":"-","title":"Polynomials4ML.sizeY","text":"sizeY(maxL): Return the size of the set of spherical harmonics Y_lm(θφ) of degree less than or equal to the given maximum degree maxL\n\n\n\n\n\n","category":"method"},{"location":"experimental/#Experimental-API","page":"Experimental","title":"Experimental API","text":"","category":"section"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"The interfaces specified below are experimental and not part of the public API yet. Some of it is not even implemented yet and are just being sketched out in separate branches. There is no guarantee that these are provided for all of the exported basis sets, and there is no guarantee of semver-compatible backward compatibility at this point.","category":"page"},{"location":"experimental/#Re-using-Basis-Output-Arrays","page":"Experimental","title":"Re-using Basis Output Arrays","text":"","category":"section"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"The default output arrays are of type CachedArray. This means that after they have been used, they can be released back into an array cache from which they have been acquired. (See ObjectPools.jl for more details.) This will avoid a new allocation next time a basis is evaluated. The interface for this is ","category":"page"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"B = evaluate(basis, X)\nrelease!(B)\nB, dB = evaluate_ed(basis, X)\nrelease!(B)\nrelease!(dB)\n# ... and so forth ... ","category":"page"},{"location":"experimental/#Laplacian","page":"Experimental","title":"Laplacian","text":"","category":"section"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"The laplacian interface is experimental and should not be considered part of the public API. ","category":"page"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"For some applications it is important to have a fast evaluation of the laplace operator, which can often be achieved at far lower computational cost than a hessian. For example, spherical harmonics are eigenfunctions of the laplacian while solid harmonics have zero-laplacian. To exploit this we provide both in-place and allocating interfaces to evaluate the laplacians. In addition we provide an interface to evaluate the basis, its gradients as well as the laplacian, analogous to evaluate_ed2 above. This interface is convenient to evaluate laplacians of chains.","category":"page"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"laplacian!(ΔY, basis, X)\nΔY = laplacian(basis, X)\neval_grad_laplace!(Y, dY, ΔY, basis, X)\nY, dY, ΔY = eval_grad_laplace(basis, X)","category":"page"},{"location":"experimental/#Backward-Differentiation-w.r.t.-Inputs-X","page":"Experimental","title":"Backward Differentiation w.r.t. Inputs X","text":"","category":"section"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"[WORK IN PROGRESS] We implement \"manual\" pullbacks w.r.t. the X variable. These  take the form","category":"page"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"∂X = pb_evaluate(basis, ∂B, X, args..)\npb_evaluate!(∂X, basis, ∂B, X, args...)","category":"page"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"and analogously for the evaluate_*** variants. The args... can differ between different basis sets e.g. may rely on intermediate results in the evaluation of the basis. ","category":"page"},{"location":"experimental/#Lux","page":"Experimental","title":"Lux","text":"","category":"section"},{"location":"experimental/","page":"Experimental","title":"Experimental","text":"[TODO]","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Polynomials4ML","category":"page"},{"location":"#Polynomials4ML","page":"Home","title":"Polynomials4ML","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Polynomials4ML.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"api.md\", \"experimental.md\", \"docstrings.md\"]\nDepth = 3","category":"page"},{"location":"docstrings/#Docstrings","page":"Docstrings","title":"Docstrings","text":"","category":"section"},{"location":"docstrings/","page":"Docstrings","title":"Docstrings","text":"CurrentModule = Polynomials4ML","category":"page"},{"location":"docstrings/","page":"Docstrings","title":"Docstrings","text":"This page lists all docstrings in Polynomials4ML including for functions that are not part of the public API. Please check with Public API which functionality is guaranteed semver-stable.","category":"page"},{"location":"docstrings/","page":"Docstrings","title":"Docstrings","text":"","category":"page"}]
}
