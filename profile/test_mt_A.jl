
using BenchmarkTools, Test, Polynomials4ML
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!, BB_prod
using StrideArrays, LoopVectorization, Tullio, Polyester
using Main.Threads

P4ML = Polynomials4ML

##

function eval_mt1!(A, basis::PooledSparseProduct, BB)
   nBB = length(BB)
   nA = length(basis) 
   nX = size(BB[1], 1) 
   @assert length(A) >= nA 
   @assert all(size(BB[i], 1) == nX for i = 2:nBB)

   B1, B2, B3 = BB      

   @threads :static for iA = 1:nA
      a = 0.0 
      @inbounds begin 
         b = basis.spec[iA]
         @simd ivdep for iX = 1:nX 
            a += B1[iX, b[1]] * B2[iX, b[2]] * B3[iX, b[3]]
         end
         A[iA] = a 
      end
   end

   return nothing 
end 

function eval_mt2!(A, basis::PooledSparseProduct, BB)
   nBB = length(BB)
   nA = length(basis) 
   nX = size(BB[1], 1) 
   @assert length(A) >= nA 
   @assert all(size(BB[i], 1) == nX for i = 2:nBB)

   b = basis.spec
   B1, B2, B3 = BB      
   @tullio A[i] = B1[a, b[i][1]] * B2[a, b[i][2]] * B3[a, b[i][3]]

   return nothing 
end 


function eval_mt3!(A, basis::PooledSparseProduct, BB)
   nBB = length(BB)
   nA = length(basis) 
   nX = size(BB[1], 1) 
   @assert length(A) >= nA 
   @assert all(size(BB[i], 1) == nX for i = 2:nBB)

   B1, B2, B3 = BB      

   @batch for iA = 1:nA
      a = 0.0 
      b = basis.spec[iA]
      @avx for iX = 1:nX 
         a += BB_prod(b, BB)
      end
      @inbounds A[iA] = a 
   end

   return nothing 
end 



##

N1 = 10
N2 = 20
N3 = 50

B1 = randn(N1)
B2 = randn(N2)
B3 = randn(N3)

spec = sort([(rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:1000])

basis = PooledSparseProduct(spec)

## 

nX = 64
_B1 = randn(nX, N1); _B2 = randn(nX, N2); _B3 = randn(nX, N3)
B1 = PtrArray(_B1); B2 = PtrArray(_B2); B3 = PtrArray(_B3)
_A0 = zeros(length(basis)); _A1 = zeros(length(basis)); _A2 = zeros(length(basis))
A0 = PtrArray(_A0); A1 = PtrArray(_A1); A2 = PtrArray(_A2)

BB = (B1, B2, B3)

evaluate!(A0, basis, BB)
eval_mt1!(A1, basis, BB)
eval_mt3!(A2, basis, BB)
A0 ≈ A1
A0 ≈ A2


##

@btime evaluate!($A0, $basis, $BB)
@btime eval_mt1!($A1, $basis, $BB)
@btime eval_mt3!($A2, $basis, $BB)
