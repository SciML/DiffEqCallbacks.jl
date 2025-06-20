# NOTE: `fmap` can handle all these cases without us defining them, but it often makes the
#       code type unstable. So we define them here to make the code type stable.
# Handle Non-Array Parameters in a Generic Fashion
"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_copyto!(y, x) = fmap(internal_copyto!, y, x; exclude = isleaf)

function internal_copyto!(y, x)
    hasmethod(copyto!, Tuple{typeof(y), typeof(x)}) ? copyto!(y, x) : nothing
end

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
recursive_neg!(x) = fmap(internal_neg!, x; exclude = isleaf)

internal_neg!(x::AbstractArray) = x .*= -1
internal_neg!(x) = nothing

"""
    zero!(x)

`x .= 0` for generic `x`. This is used to handle non-array parameters!
"""
recursive_zero!(x) = fmap(internal_zero!, x; exclude = isleaf)

internal_zero!(x::AbstractArray) = fill!(x, false)
internal_zero!(x) = nothing
"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_sub!(y, x) = fmap(internal_sub!, y, x; exclude = isleaf)

internal_sub!(y::AbstractArray, x::AbstractArray) = y .-= x
internal_sub!(y, x) = nothing
"""
    recursive_sub(y, x)

`y .- x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_sub(y, x) = fmap(internal_sub, y, x; exclude = isleaf)

internal_sub(y::AbstractArray, x::AbstractArray) = y .- x
internal_sub(y, x) = nothing
"""
    recursive_add!(y, x)

`y .+= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_add!(y, x) = fmap(internal_add!, y, x; exclude = isleaf)

internal_add!(y::AbstractArray, x::AbstractArray) = y .+= x
internal_add!(y, x) = nothing
"""
    recursive_sum(x)

`sum(x)` for generic `x`. This is used to handle non-array parameters!
"""
recursive_sum(x) = fmap(internal_sum, x; exclude=isleaf)

internal_sum(x::AbstractArray) = sum(x)
internal_sum(x) = nothing
"""
    recursive_mul(x,y)

`y .* x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_mul(x,y) = fmap(internal_mul, y, x; exclude = isleaf)

internal_mul(x::AbstractArray, y::AbstractArray) = x.*y
internal_mul(x,y) = nothing
"""
    recursive_abs(x)

`abs.(x)` for generic `x`. This is used to handle non-array parameters!
"""
recursive_abs(x) = fmap(internal_abs, x; exclude=isleaf)

internal_abs(x::AbstractArray) = abs.(x)
internal_abs(x) = nothing
"""
    allocate_vjp(λ, x)
    allocate_vjp(x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
allocate_vjp(λ::AbstractArray, x) = fmap(
    Base.Fix1(allocate_vjp_internal, λ), x; exclude = isleaf)
allocate_vjp(x) = fmap(similar, x)

allocate_vjp_internal(λ::AbstractArray, x) = similar(λ, size(x))

"""
    allocate_zeros(x)

`zero.(x)` for generic `x`. This is used to handle non-array parameters!
"""
allocate_zeros(x) = fmap(internal_allocate_zeros, x; exclude = isleaf)

internal_allocate_zeros(x) = hasmethod(zero, Tuple{typeof(x)}) ? zero(x) : nothing

"""
recursive_copy(y)

`copy(y)` for generic `y`. This is used to handle non-array parameters!
"""
recursive_copy(y) = fmap(internal_copy, y; exclude = isleaf)

internal_copy(x) = hasmethod(copy, Tuple{typeof(x)}) ? copy(x) : nothing

"""
    recursive_adjoint(y)

`adjoint(y)` for generic `y`. This is used to handle non-array parameters!
"""
recursive_adjoint(y) = fmap(internal_adjoint, y; exclude = isleaf)

internal_adjoint(x) = hasmethod(adjoint, Tuple{typeof(x)}) ? adjoint(x) : nothing

# scalar_mul!
recursive_scalar_mul!(x, α) = fmap(Base.Fix2(internal_scalar_mul!, α), x; exclude = isleaf)

internal_scalar_mul!(x::Number, α) = x * α
internal_scalar_mul!(x::AbstractArray, α) = x .*= α
internal_scalar_mul!(x, α) = nothing

# axpy!
function recursive_axpy!(α, x, y)
    fmap((xᵢ, yᵢ) -> internal_axpy!(α, xᵢ, yᵢ), x, y; exclude = isleaf)
end

internal_axpy!(α, x::AbstractArray, y::AbstractArray) = axpy!(α, x, y)
internal_axpy!(α, x, y) = nothing

# isleaf
isleaf(x) = Functors.isleaf(x)

## BigFloat and such are not bitstype
isleaf(::AbstractArray{T}) where {T} = isbitstype(T) || T <: Number
