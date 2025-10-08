"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
function recursive_copyto! end

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
function recursive_neg! end
"""
    zero!(x)

`x .= 0` for generic `x`. This is used to handle non-array parameters!
"""
function recursive_zero! end

"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
function recursive_sub! end

"""
    recursive_add!(y, x)

`y .+= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
function recursive_add! end

"""
    allocate_vjp(λ, x)
    allocate_vjp(x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
function allocate_vjp end

"""
    allocate_zeros(x)

`zero.(x)` for generic `x`. This is used to handle non-array parameters!
"""
function allocate_zeros end

"""
recursive_copy(y)

`copy(y)` for generic `y`. This is used to handle non-array parameters!
"""
function recursive_copy end

"""
    recursive_adjoint(y)

`adjoint(y)` for generic `y`. This is used to handle non-array parameters!
"""
function recursive_adjoint end

function recursive_scalar_mul! end

function recursive_axpy! end
