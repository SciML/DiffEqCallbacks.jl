mutable struct StepsizeLimiterAffect{F,T,T2,T3}
    dtFE::F
    cached_dtcache::T
    safety_factor::T2
    max_step::T3
end
# Now make `affect!` for this:
function (p::StepsizeLimiterAffect)(integrator)
    integrator.opts.dtmax =
        p.safety_factor * p.dtFE(integrator.u, integrator.p, integrator.t)
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

function StepsizeLimiter(
    dtFE;
    safety_factor = 9 // 10,
    max_step = false,
    cached_dtcache = 0.0,
)
    affect! = StepsizeLimiterAffect(dtFE, cached_dtcache, safety_factor, max_step)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(
        condtion,
        affect!;
        initialize = StepsizeLimiter_initialize,
        save_positions = (false, false),
    )
end

export StepsizeLimiter
