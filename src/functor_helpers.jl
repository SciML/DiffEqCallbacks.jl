# NOTE: `fmap` can handle all these cases without us defining them, but it often makes the
#       code type unstable. So we define them here to make the code type stable.
# Handle Non-Array Parameters in a Generic Fashion
"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_copyto!(y, x) = fmap(internal_copyto!, y, x)

function internal_copyto!(y, x)
    hasmethod(copyto!, Tuple{typeof(y), typeof(x)}) ? copyto!(y, x) : nothing
end

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
recursive_neg!(x) = fmap(internal_neg!, x)

internal_neg!(x::Number) = -x
internal_neg!(x::AbstractArray) = x .*= -1
internal_neg!(x) = nothing

"""
    zero!(x)

`x .= 0` for generic `x`. This is used to handle non-array parameters!
"""
recursive_zero!(x) = fmap(internal_zero!, x)

internal_zero!(x::Number) = zero(x)
internal_zero!(x::AbstractArray) = fill!(x, zero(eltype(x)))
internal_zero!(x) = nothing

"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_sub!(y, x) = fmap(internal_sub!, y, x)

internal_sub!(x::Number, y::Number) = x - y
internal_sub!(x::AbstractArray, y::AbstractArray) = axpy!(-1, y, x)
internal_sub!(x, y) = nothing

"""
    recursive_add!(y, x)

`y .+= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_add!(y, x) = fmap(internal_add!, y, x)

internal_add!(x::Number, y::Number) = x + y
internal_add!(x::AbstractArray, y::AbstractArray) = y .+= x
internal_add!(x, y) = nothing

"""
    allocate_vjp(λ, x)
    allocate_vjp(x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
allocate_vjp(λ::AbstractArray, x::AbstractArray) = similar(λ, size(x))
allocate_vjp(λ::AbstractArray, x::Tuple) = allocate_vjp.((λ,), x)
function allocate_vjp(λ::AbstractArray, x::NamedTuple{F}) where {F}
    NamedTuple{F}(allocate_vjp.((λ,), values(x)))
end
allocate_vjp(λ::AbstractArray, x) = fmap(Base.Fix1(allocate_vjp, λ), x)

allocate_vjp(x::AbstractArray) = similar(x)
allocate_vjp(x::Tuple) = allocate_vjp.(x)
allocate_vjp(x::NamedTuple{F}) where {F} = NamedTuple{F}(allocate_vjp.(values(x)))
allocate_vjp(x) = fmap(allocate_vjp, x)

"""
    allocate_zeros(x)

`zero.(x)` for generic `x`. This is used to handle non-array parameters!
"""
allocate_zeros(x) = fmap(internal_allocate_zeros, x)

internal_allocate_zeros(x) = hasmethod(zero, Tuple{typeof(x)}) ? zero(x) : nothing

"""
recursive_copy(y)

`copy(y)` for generic `y`. This is used to handle non-array parameters!
"""
recursive_copy(y) = fmap(internal_copy, y)

internal_copy(x) = hasmethod(copy, Tuple{typeof(x)}) ? copy(x) : nothing

"""
    recursive_adjoint(y)

`adjoint(y)` for generic `y`. This is used to handle non-array parameters!
"""
recursive_adjoint(y) = fmap(internal_adjoint, y)

internal_adjoint(x) = hasmethod(adjoint, Tuple{typeof(x)}) ? adjoint(x) : nothing

# scalar_mul!
recursive_scalar_mul!(x, α) = fmap(Base.Fix2(internal_scalar_mul!, α), x)

internal_scalar_mul!(x::Number, α) = x * α
internal_scalar_mul!(x::AbstractArray, α) = x .*= α
internal_scalar_mul!(x, α) = nothing

# axpy!
recursive_axpy!(α, x, y) = fmap((xᵢ, yᵢ) -> internal_axpy!(α, xᵢ, yᵢ), x, y)

internal_axpy!(α, x::Number, y::Number) = y + α * x
internal_axpy!(α, x::AbstractArray, y::AbstractArray) = axpy!(α, x, y)
internal_axpy!(α, x, y) = nothing
