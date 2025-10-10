module DiffEqCallbacksFunctorsExt

import DiffEqCallbacks: recursive_copyto!, recursive_neg!, recursive_zero!, recursive_sub!,
                        recursive_add!, allocate_vjp, allocate_zeros, recursive_copy,
                        recursive_adjoint, recursive_scalar_mul!, recursive_axpy!,
                        internal_copyto!, internal_neg!, internal_zero!, internal_sub!,
                        internal_add!, allocate_vjp_internal, internal_allocate_zeros,
                        internal_copy, internal_adjoint, internal_scalar_mul!,
                        internal_axpy!
using Functors: Functors, fmap

# NOTE: `fmap` can handle all these cases without us defining them, but it often makes the
#       code type unstable. So we define them here to make the code type stable.
# Handle Non-Array Parameters in a Generic Fashion
recursive_copyto!(y, x) = fmap(internal_copyto!, y, x; exclude = isleaf)

recursive_neg!(x) = fmap(internal_neg!, x; exclude = isleaf)

recursive_zero!(x) = fmap(internal_zero!, x; exclude = isleaf)

recursive_sub!(y, x) = fmap(internal_sub!, y, x; exclude = isleaf)

recursive_add!(y, x) = fmap(internal_add!, y, x; exclude = isleaf)

function allocate_vjp(λ, x)
    fmap(
        Base.Fix1(allocate_vjp_internal, λ), x; exclude = isleaf)
end
allocate_vjp(x) = fmap(similar, x)

allocate_zeros(x) = fmap(internal_allocate_zeros, x; exclude = isleaf)

recursive_copy(y) = fmap(internal_copy, y; exclude = isleaf)

recursive_adjoint(y) = fmap(internal_adjoint, y; exclude = isleaf)

# scalar_mul!
recursive_scalar_mul!(x, α) = fmap(Base.Fix2(internal_scalar_mul!, α), x; exclude = isleaf)

# axpy!
function recursive_axpy!(α, x, y)
    fmap((xᵢ, yᵢ) -> internal_axpy!(α, xᵢ, yᵢ), x, y; exclude = isleaf)
end

# isleaf
isleaf(x) = Functors.isleaf(x)

## BigFloat and such are not bitstype
isleaf(::AbstractArray{T}) where {T} = isbitstype(T) || T <: Number

end
