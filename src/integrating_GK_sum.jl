mutable struct SavingIntegrandGKSumAffect{
        IntegrandFunc,
        integrandType,
        IntegrandCacheType,
    }
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValuesSum{integrandType}
    integrand_cache::IntegrandCacheType
    accumulation_cache::IntegrandCacheType
    gk_step_cache::IntegrandCacheType
    gk_err_cache::IntegrandCacheType
    tol::Float64
    integrand_inplace::Union{Nothing, Bool}
end

function integrate_gk!(
        affect!::SavingIntegrandGKSumAffect, integrator,
        bound_l, bound_r; order = 7, tol = 1.0e-7
    )
    affect!.gk_step_cache = recursive_zero!(affect!.gk_step_cache)
    affect!.gk_err_cache = recursive_zero!(affect!.gk_err_cache)
    isinplace_prob = SciMLBase.isinplace(integrator.sol.prob)
    inplace_integrand = affect!.integrand_inplace === nothing ?
        (isinplace_prob && affect!.integrand_cache !== nothing) :
        affect!.integrand_inplace
    for i in 1:(2 * order + 1)
        t_temp = (gk_points[order][i] + 1) * ((bound_r - bound_l) / 2) + bound_l
        if isinplace_prob
            curu = first(get_tmp_cache(integrator))
            integrator(curu, t_temp)
        else
            curu = integrator(t_temp)
        end
        if inplace_integrand
            affect!.integrand_func(affect!.integrand_cache, curu, t_temp, integrator)
            recursive_axpy!(
                gk_weights[order][i],
                affect!.integrand_cache, affect!.gk_step_cache
            )
            if i % 2 == 0
                recursive_axpy!(
                    g_weights[order][div(i, 2)],
                    affect!.integrand_cache, affect!.gk_err_cache
                )
            end
        else
            recursive_axpy!(
                gk_weights[order][i],
                affect!.integrand_func(curu, t_temp, integrator), affect!.gk_step_cache
            )
            if i % 2 == 0
                recursive_axpy!(
                    g_weights[order][div(i, 2)],
                    affect!.integrand_func(curu, t_temp, integrator), affect!.gk_err_cache
                )
            end
        end
    end
    return if sum(abs.((affect!.gk_step_cache .- affect!.gk_err_cache) .* (bound_r - bound_l) ./ 2)) < tol
        recursive_axpy!(1, affect!.gk_step_cache .* (bound_r - bound_l) ./ 2, affect!.accumulation_cache)
    else
        integrate_gk!(
            affect!, integrator, bound_l, (bound_l + bound_r) / 2, order = order, tol = tol / 2
        )
        integrate_gk!(
            affect!, integrator, (bound_l + bound_r) / 2, bound_r, order = order, tol = tol / 2
        )
    end
end

function (affect!::SavingIntegrandGKSumAffect)(integrator)
    n = 0
    if integrator.sol.prob isa Union{SDEProblem, RODEProblem}
        throw("Gauss-Kronrod algorithm is not necessarily convergent for this problem type")
    else
        n = div(SciMLBase.alg_order(integrator.alg) + 1, 2)
    end
    accumulation_cache = recursive_zero!(affect!.accumulation_cache)
    integrate_gk!(
        affect!, integrator, integrator.tprev, integrator.t, order = n, tol = affect!.tol
    )
    recursive_add!(affect!.integrand_values.integrand, accumulation_cache)
    return derivative_discontinuity!(integrator, false)
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

## Keyword Arguments

  - `integrand_inplace = nothing`: controls which form of `integrand_func` is called.
    With the default `nothing`, the in-place `integrand_func(out, u, t, integrator)`
    form is used for in-place problems (when an `integrand_prototype` is given) and
    the allocating `integrand_func(u, t, integrator)` form for out-of-place problems.
    Pass `integrand_inplace = true` to force the in-place form even for out-of-place
    problems — the integrand output (e.g. a parameter-shaped buffer) may be mutable
    even when the state is immutable, which avoids allocating the output on every
    quadrature node. Requires an `integrand_prototype`. Pass `integrand_inplace = false`
    to force the allocating form.

The outputted values are saved into `integrand_values`. The values are found
via `integrand_values.integrand`.

!!! note

    This method uses Gauss-Kronrod quadrature rule to allow for error control.

    This method is currently limited to ODE solvers of order 10 or lower. Open an issue if other
    solvers are required.
"""
function IntegratingGKSumCallback(
        integrand_func, integrand_values::IntegrandValuesSum, integrand_prototype,
        tol = 1.0e-7;
        integrand_inplace::Union{Nothing, Bool} = nothing
    )
    if integrand_inplace === true && integrand_prototype === nothing
        throw(
            ArgumentError(
                "integrand_inplace = true requires an integrand_prototype to use as the output buffer."
            )
        )
    end
    affect! = SavingIntegrandGKSumAffect(
        integrand_func, integrand_values, integrand_prototype,
        allocate_zeros(integrand_prototype), allocate_zeros(integrand_prototype),
        allocate_zeros(integrand_prototype), tol, integrand_inplace
    )
    condition = true_condition
    return DiscreteCallback(condition, affect!, save_positions = (false, false))
end

export IntegratingGKSumCallback
