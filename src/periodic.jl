function PeriodicCallback(f, Δt::Number; initialize = DiffEqBase.INITIALIZE_DEFAULT, kwargs...)
    @assert Δt > 0

    # Value of `t` at which `f` should be called next:
    tnext = Ref(typemax(Δt))
    condition = (t, u, integrator) -> t == tnext[]

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        f(integrator)

        # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
        tnew = tnext[] + Δt
        tstops = integrator.opts.tstops
        for i in length(tstops) : -1 : 1 # reverse iterate to encounter large elements earlier
            if tstops.valtree[i] > tnew # TODO: relying on implementation detail
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            end
        end
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, t, u, integrator)
        initialize(c, t, u, integrator)
        tnext[] = t
        affect!(integrator)
    end

    DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

export PeriodicCallback
