"""
    IntegrandValuesSum{integrandType}

A struct used to save values of the integrand values in `integrand::Vector{integrandType}`.
"""
mutable struct IntegrandValuesSum{integrandType}
    integrand::integrandType
end

"""
    IntegrandValuesSum(integrandType::DataType)

Return `IntegrandValuesSum{integrandType}` with empty storage vectors.
"""
function IntegrandValuesSum(::Type{integrandType}) where {integrandType}
    IntegrandValuesSum{integrandType}(integrandType)
end

function Base.show(io::IO, integrand_values::IntegrandValuesSum)
    integrandType = eltype(integrand_values.integrand)
    print(io, "IntegrandValuesSum{integrandType=", integrandType, "}",
        "\nintegrand:\n", integrand_values.integrand)
end

mutable struct SavingIntegrandSumAffect{IntegrandFunc, integrandType, IntegrandCacheType}
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValuesSum{integrandType}
    integrand_cache::IntegrandCacheType
    accumulation_cache::IntegrandCacheType
end

function (affect!::SavingIntegrandSumAffect)(integrator)
    n = 0
    if integrator.sol.prob isa Union{SDEProblem, RODEProblem}
        n = 10
    else
        n = div(SciMLBase.alg_order(integrator.alg) + 1, 2)
    end
    accumulation_cache = recursive_zero!(affect!.accumulation_cache)
    for i in 1:n
        t_temp = ((integrator.t - integrator.tprev) / 2) * gauss_points[n][i] +
                 (integrator.t + integrator.tprev) / 2
        if DiffEqBase.isinplace(integrator.sol.prob)
            curu = first(get_tmp_cache(integrator))
            integrator(curu, t_temp)
            if affect!.integrand_cache == nothing
                recursive_axpy!(gauss_weights[n][i],
                    affect!.integrand_func(curu, t_temp, integrator), accumulation_cache)
            else
                affect!.integrand_func(affect!.integrand_cache, curu, t_temp, integrator)
                recursive_axpy!(
                    gauss_weights[n][i], affect!.integrand_cache, accumulation_cache)
            end
        else
            recursive_axpy!(gauss_weights[n][i],
                affect!.integrand_func(integrator(t_temp), t_temp, integrator), accumulation_cache)
        end
    end
    recursive_scalar_mul!(accumulation_cache, (integrator.t - integrator.tprev) / 2)
    recursive_add!(affect!.integrand_values.integrand, accumulation_cache)
    u_modified!(integrator, false)
end

"""
```julia
IntegratingCallback(integrand_func,
    integrand_values::IntegrandValues,
    cache = nothing)
```

Lets one define a function `integrand_func(u, t, integrator)` which
returns Integral(integrand_func(u(t),t)dt over the problem tspan.

## Arguments

  - `integrand_func(out, u, t, integrator)` for in-place problems and `out = integrand_func(u, t, integrator)` for
    out-of-place problems. Returns the quantity in the integral for computing dG/dp.
    Note that for out-of-place problems, this should allocate the output (not as a view to `u`).
  - `integrand_values::IntegrandValues` is the types that `integrand_func` will return, i.e.
    `integrand_func(t, u, integrator)::integrandType`. It's specified via
    `IntegrandValues(integrandType)`, i.e. give the type
    that `integrand_func` will output (or higher compatible type).
  - `integrand_prototype` is a prototype of the output from the integrand.

The outputted values are saved into `integrand_values`. The values are found
via `integrand_values.integrand`.

!!! note

    This method is currently limited to ODE solvers of order 10 or lower. Open an issue if other
    solvers are required.
"""
function IntegratingSumCallback(
        integrand_func, integrand_values::IntegrandValuesSum, integrand_prototype)
    affect! = SavingIntegrandSumAffect(
        integrand_func, integrand_values, integrand_prototype,
        allocate_zeros(integrand_prototype))
    condition = (u, t, integrator) -> true
    DiscreteCallback(condition, affect!, save_positions = (false, false))
end

export IntegratingSumCallback, IntegrandValuesSum
