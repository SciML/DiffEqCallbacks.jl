function PeriodicCallback(f, Δt::Number; kwargs...)
    @assert Δt > 0

    # Value of `t` at which `f` should be called next:
    t_next = Ref(typemax(Δt))
    condition = (t, u, integrator) -> t == t_next[]

    # Call f, update t_next, and make sure we stop at the new t_next
    affect! = function (integrator)
        f(integrator)

        # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
        t_new = t_next[] + Δt
        if any(t -> t > t_new, integrator.opts.tstops.valtree) # TODO: accessing internal data...
            t_next[] = t_new
            add_tstop!(integrator, t_new)
        end
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize = function (c, t, u, integrator)
        t_next[] = t
        affect!(integrator)
    end

    DiscreteCallback(condition, affect!; initialize = initialize, kwargs...)
end

export PeriodicCallback
