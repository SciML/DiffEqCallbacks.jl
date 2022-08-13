mutable struct StepsizeLimiterAffect{F, T, T2, T3}
    dtFE::F
    cached_dtcache::T
    safety_factor::T2
    max_step::T3
end
# Now make `affect!` for this:
function (p::StepsizeLimiterAffect)(integrator)
    integrator.opts.dtmax = p.safety_factor *
                            p.dtFE(integrator.u, integrator.p, integrator.t)
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
    u_modified!(integrator, false)
end

function StepsizeLimiter_initialize(cb, u, t, integrator)
    cb.affect!.cached_dtcache = integrator.dtcache
    cb.affect!(integrator)
end

"""
```julia
StepsizeLimiter(dtFE;safety_factor=9//10,max_step=false,cached_dtcache=0.0)
```

In many cases there is a known maximal stepsize for which the computation is
stable and produces correct results. For example, in hyperbolic PDEs one normally
needs to ensure that the stepsize stays below some ``\Delta t_{FE}`` determined
by the CFL condition. For nonlinear hyperbolic PDEs this limit can be a function
`dtFE(u,p,t)` which changes throughout the computation. The stepsize limiter lets
you pass a function which will adaptively limit the stepsizes to match these
constraints.

## Arguments

- `dtFE` is the maximal timestep and is calculated using the previous `t` and `u`.

## Keyword Arguments

- `safety_factor` is the factor below the true maximum that will be stepped to
  which defaults to `9//10`. `max_step=true` makes every step equal to
- `safety_factor*dtFE(u,p,t)` when the solver is set to `adaptive=false`. `cached_dtcache`
  should be set to match the type for time when not using Float64 values.
"""
function StepsizeLimiter(dtFE; safety_factor = 9 // 10, max_step = false,
                         cached_dtcache = 0.0)
    affect! = StepsizeLimiterAffect(dtFE, cached_dtcache, safety_factor, max_step)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(condtion, affect!;
                     initialize = StepsizeLimiter_initialize,
                     save_positions = (false, false))
end

export StepsizeLimiter
