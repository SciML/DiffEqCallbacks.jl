"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
function recursive_copyto! end

recursive_copyto!(y::AbstractArray, x::AbstractArray) = internal_copyto!(y, x)

function internal_copyto!(y, x)
    return hasmethod(copyto!, Tuple{typeof(y), typeof(x)}) ? copyto!(y, x) : nothing
end

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
function recursive_neg! end

recursive_neg!(x::AbstractArray) = internal_neg!(x)

internal_neg!(x::AbstractArray) = x .*= -1
internal_neg!(x) = nothing

"""
    zero!(x)

`x .= 0` for generic `x`. This is used to handle non-array parameters!
"""
function recursive_zero! end

recursive_zero!(x::AbstractArray) = internal_zero!(x)
recursive_zero!(x::Number) = zero(x)
recursive_zero!(::Nothing) = nothing

internal_zero!(x::AbstractArray) = fill!(x, false)
internal_zero!(x) = nothing

"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
function recursive_sub! end

recursive_sub!(y::AbstractArray, x::AbstractArray) = internal_sub!(y, x)

internal_sub!(y::AbstractArray, x::AbstractArray) = y .-= x
internal_sub!(y, x) = nothing

"""
    recursive_add!(y, x)

`y .+= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
function recursive_add! end

recursive_add!(y::AbstractArray, x::AbstractArray) = internal_add!(y, x)

internal_add!(y::AbstractArray, x::AbstractArray) = y .+= x
internal_add!(y, x) = nothing

"""
    allocate_vjp(λ, x)
    allocate_vjp(x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
function allocate_vjp end

allocate_vjp(λ::AbstractArray, x) = allocate_vjp_internal(λ, x)

allocate_vjp_internal(λ::AbstractArray, x) = similar(λ, size(x))

"""
    allocate_zeros(x)

`zero.(x)` for generic `x`. This is used to handle non-array parameters!
"""
function allocate_zeros end

allocate_zeros(x::AbstractArray) = internal_allocate_zeros(x)
allocate_zeros(x::Number) = zero(x)
allocate_zeros(::Nothing) = nothing

internal_allocate_zeros(x) = hasmethod(zero, Tuple{typeof(x)}) ? zero(x) : nothing

"""
recursive_copy(y)

`copy(y)` for generic `y`. This is used to handle non-array parameters!
"""
function recursive_copy end

recursive_copy(x::AbstractArray) = internal_copy(x)
recursive_copy(x::Number) = x
recursive_copy(::Nothing) = nothing

internal_copy(x) = hasmethod(copy, Tuple{typeof(x)}) ? copy(x) : nothing

"""
    recursive_adjoint(y)

`adjoint(y)` for generic `y`. This is used to handle non-array parameters!
"""
function recursive_adjoint end

recursive_adjoint(x::AbstractArray) = internal_adjoint(x)

internal_adjoint(x) = hasmethod(adjoint, Tuple{typeof(x)}) ? adjoint(x) : nothing

function recursive_scalar_mul! end

recursive_scalar_mul!(x::AbstractArray, α) = internal_scalar_mul!(x, α)
recursive_scalar_mul!(x::Number, α) = x * α
recursive_scalar_mul!(::Nothing, α) = nothing

internal_scalar_mul!(x::Number, α) = x * α
internal_scalar_mul!(x::AbstractArray, α) = x .*= α
internal_scalar_mul!(x, α) = nothing

function recursive_axpy! end

recursive_axpy!(α, x::AbstractArray, y::AbstractArray) = internal_axpy!(α, x, y)
recursive_axpy!(α, x::Number, y::Number) = y + α * x
# For out-of-place integrands with nothing as accumulator, start accumulation
recursive_axpy!(α, x, ::Nothing) = α * x

internal_axpy!(α, x::AbstractArray, y::AbstractArray) = axpy!(α, x, y)
internal_axpy!(α, x, y) = nothing
