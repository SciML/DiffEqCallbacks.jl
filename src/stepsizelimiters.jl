mutable struct StepsizeLimiterAffect{F, T, T2, T3}
    dtFE::F
    cached_dtcache::T
    safety_factor::T2
    max_step::T3
end
# Now make `affect!` for this:
function (p::StepsizeLimiterAffect)(integrator)
    integrator.opts.dtmax = p.safety_factor * p.dtFE(integrator.u, integrator.p, integrator.t)
    if !integrator.opts.adaptive
        if integrator.opts.dtmax < integrator.dtcache
            integrator.dtcache = integrator.opts.dtmax
        elseif p.cached_dtcache <= integrator.opts.dtmax
            integrator.dtcache = p.cached_dtcache
        end
    end
    if p.max_step
        set_proposed_dt!(integrator, integrator.opts.dtmax)
        integrator.dtcache = integrator.opts.dtmax
    end
    return derivative_discontinuity!(integrator, false)
end

function StepsizeLimiter_initialize(cb, u, t, integrator)
    cb.affect!.cached_dtcache = integrator.dtcache
    return cb.affect!(integrator)
end

"""
    StepsizeLimiter(dtFE; safety_factor = 9 // 10, max_step = false,
        cached_dtcache = 0.0) -> DiscreteCallback

In many cases, there is a known maximal stepsize for which the computation is
stable and produces correct results. For example, in hyperbolic PDEs one normally
needs to ensure that the stepsize stays below some ``\\Delta t_{FE}`` determined
by the CFL condition. For nonlinear hyperbolic PDEs this limit can be a function
`dtFE(u,p,t)` which changes throughout the computation. The stepsize limiter lets
you pass a function which will adaptively limit the stepsizes to match these
constraints.

# Arguments

  - `dtFE`: function called as `dtFE(u, p, t)` to compute the current maximum stable step.

# Keywords

  - `safety_factor = 9 // 10`: factor applied to the maximum returned by `dtFE`.
  - `max_step::Bool = false`: when `true`, set every proposed step to
    `safety_factor * dtFE(u, p, t)`, including for a non-adaptive solver.
  - `cached_dtcache = 0.0`: initial cache for the unconstrained step. Set it to a value with
    the problem time type when that type is not `Float64`.

# Returns

  - `DiscreteCallback`: a callback that updates `integrator.opts.dtmax` before every step.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

f(u, p, t) = -u
prob = ODEProblem(f, 1.0, (0.0, 1.0))
dtFE(u, p, t) = 0.05
cb = StepsizeLimiter(dtFE; safety_factor = 0.8)

sol = solve(prob, Tsit5(); callback = cb)
```
"""
function StepsizeLimiter(
        dtFE; safety_factor = 9 // 10, max_step = false,
        cached_dtcache = 0.0
    )
    affect! = StepsizeLimiterAffect(dtFE, cached_dtcache, safety_factor, max_step)
    condition = true_condition
    return DiscreteCallback(
        condition, affect!;
        initialize = StepsizeLimiter_initialize,
        save_positions = (false, false)
    )
end

export StepsizeLimiter
