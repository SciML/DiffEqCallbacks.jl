mutable struct SavingIntegrandGKSumAffect{
        IntegrandFunc,
        integrandType,
        IntegrandCacheType,
        tolType,
        QuadratureType,
    }
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValuesSum{integrandType}
    integrand_cache::IntegrandCacheType
    accumulation_cache::IntegrandCacheType
    gk_step_cache::IntegrandCacheType
    gk_err_cache::IntegrandCacheType
    tol::tolType
    gk_pts::Vector{Vector{QuadratureType}}
    gk_ws::Vector{Vector{QuadratureType}}
    g_ws::Vector{Vector{QuadratureType}}
    integrand_inplace::Union{Nothing, Bool}
    tol_warned::Bool
end

function integrate_gk!(
        affect!::SavingIntegrandGKSumAffect, integrator,
        bound_l, bound_r; order = 7, tol = 1.0e-7, depth = 0
    )
    affect!.gk_step_cache = recursive_zero!(affect!.gk_step_cache)
    affect!.gk_err_cache = recursive_zero!(affect!.gk_err_cache)
    isinplace_prob = SciMLBase.isinplace(integrator.sol.prob)
    inplace_integrand = affect!.integrand_inplace === nothing ?
        (isinplace_prob && affect!.integrand_cache !== nothing) :
        affect!.integrand_inplace
    for i in 1:(2 * order + 1)
        t_temp = (affect!.gk_pts[order][i] + 1) * ((bound_r - bound_l) / 2) + bound_l
        if isinplace_prob
            curu = first(get_tmp_cache(integrator))
            integrator(curu, t_temp)
        else
            curu = integrator(t_temp)
        end
        if inplace_integrand
            affect!.integrand_func(affect!.integrand_cache, curu, t_temp, integrator)
            affect!.gk_step_cache = recursive_axpy!(
                affect!.gk_ws[order][i],
                affect!.integrand_cache, affect!.gk_step_cache
            )
            if i % 2 == 0
                affect!.gk_err_cache = recursive_axpy!(
                    affect!.g_ws[order][div(i, 2)],
                    affect!.integrand_cache, affect!.gk_err_cache
                )
            end
        else
            affect!.gk_step_cache = recursive_axpy!(
                affect!.gk_ws[order][i],
                affect!.integrand_func(curu, t_temp, integrator), affect!.gk_step_cache
            )
            if i % 2 == 0
                affect!.gk_err_cache = recursive_axpy!(
                    affect!.g_ws[order][div(i, 2)],
                    affect!.integrand_func(curu, t_temp, integrator), affect!.gk_err_cache
                )
            end
        end
    end
    err_est = sum(
        abs.((affect!.gk_step_cache .- affect!.gk_err_cache) .* (bound_r - bound_l) ./ 2)
    )
    bound_m = (bound_l + bound_r) / 2
    if err_est > tol && depth < GK_MAX_DEPTH && bound_m != bound_l && bound_m != bound_r &&
            gk_has_resolvable_error(
            affect!.gk_step_cache, affect!.gk_err_cache, (bound_r - bound_l) / 2
        )
        integrate_gk!(
            affect!, integrator, bound_l, bound_m,
            order = order, tol = tol / 2, depth = depth + 1
        )
        integrate_gk!(
            affect!, integrator, bound_m, bound_r,
            order = order, tol = tol / 2, depth = depth + 1
        )
        return nothing
    end
    if err_est > tol
        warn_gk_tol_unmet!(affect!)
    end
    affect!.accumulation_cache = recursive_axpy!(
        1, affect!.gk_step_cache .* (bound_r - bound_l) ./ 2, affect!.accumulation_cache
    )
    return nothing
end

function (affect!::SavingIntegrandGKSumAffect)(integrator)
    n = 0
    if integrator.sol.prob isa Union{SDEProblem, RODEProblem}
        throw("Gauss-Kronrod algorithm is not necessarily convergent for this problem type")
    else
        n = div(SciMLBase.alg_order(integrator.alg) + 1, 2)
    end
    affect!.accumulation_cache = recursive_zero!(affect!.accumulation_cache)
    integrate_gk!(
        affect!, integrator, integrator.tprev, integrator.t, order = n, tol = affect!.tol
    )
    affect!.integrand_values.integrand = recursive_add!(
        affect!.integrand_values.integrand, affect!.accumulation_cache
    )
    return derivative_discontinuity!(integrator, false)
end

function IntegratingGKSum_initialize(cb, u, t, integrator)
    cb.affect!.tol_warned = false
    return INITIALIZE_DEFAULT(cb, u, t, integrator)
end

"""
    IntegratingGKSumCallback(integrand_func, integrand_values::IntegrandValuesSum,
        integrand_prototype, tol = 1.0e-7; integrand_inplace = nothing) -> DiscreteCallback

Construct a callback that uses adaptive Gauss-Kronrod quadrature to accumulate the integral
of `integrand_func` over accepted solver steps in `integrand_values.integrand`. Quadrature
nodes and weights are evaluated in the element type of the integrand, so higher-precision
integrands such as `BigFloat` are not capped at `Float64` precision.

# Arguments

  - `integrand_func`: define either `integrand_func(u, t, integrator)` to return the
    integrand or `integrand_func(out, u, t, integrator)` to write it into `out`. Returned or
    written values must be compatible with `integrand_values.integrand`.
  - `integrand_values::IntegrandValuesSum`: storage for the running integral. Construct it
    as
    `IntegrandValuesSum(initial_value)` with an initial value compatible with the integrand.
  - `integrand_prototype`: representative integrand output used as an in-place output
    buffer.
  - `tol::Real = 1.0e-7`: absolute error tolerance for adaptive quadrature on each accepted
    solver step. A tolerance below the floating-point roundoff level of every integrand
    component cannot be reached by further subdivision: subdivision stops at that level, a
    warning states that the tolerance was not met, and the best estimate is used.

# Keywords

  - `integrand_inplace::Union{Nothing, Bool} = nothing`: select the integrand calling form.
    With `nothing`, use the in-place form for an in-place problem when a cache can be
    allocated, and otherwise use the allocating form. Set this to `true` to force the
    in-place form or `false` to force the allocating form. `true` requires a non-`nothing`
    `integrand_prototype`.

# Returns

  - `DiscreteCallback`: a callback that adds each Gauss-Kronrod estimate to
    `integrand_values.integrand`.

# Throws

  - `ArgumentError`: if `integrand_inplace = true` and `integrand_prototype === nothing`.
  - An exception when used with an `SDEProblem` or `RODEProblem`, for which this
    Gauss-Kronrod algorithm is not guaranteed to converge.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
values = IntegrandValuesSum(0.0)
cb = IntegratingGKSumCallback((u, t, integrator) -> u^2, values, 0.0)

sol = solve(prob, Tsit5(); callback = cb)
total = values.integrand
```

!!! note

    This method uses Gauss-Kronrod quadrature rule to allow for error control.

    This method is currently limited to ODE solvers of order 10 or lower.
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
    integrand_el = eltype(eltype(integrand_values.integrand))
    prototype_el = integrand_prototype === nothing ? integrand_el : eltype(integrand_prototype)
    quadrature_el = _quadrature_eltype(integrand_el, prototype_el)
    gk_pts, gk_ws, g_ws = _gk_tables(quadrature_el)
    affect! = SavingIntegrandGKSumAffect(
        integrand_func, integrand_values, integrand_prototype,
        allocate_zeros(integrand_prototype), allocate_zeros(integrand_prototype),
        allocate_zeros(integrand_prototype), tol, gk_pts, gk_ws, g_ws, integrand_inplace, false
    )
    condition = true_condition
    return DiscreteCallback(
        condition, affect!, save_positions = (false, false),
        initialize = IntegratingGKSum_initialize
    )
end

export IntegratingGKSumCallback
