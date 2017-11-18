function IterativeCallback(time_choice, user_affect!,tType = Float64;
                           initialize = DiffEqBase.INITIALIZE_DEFAULT,
                           initial_affect = false, kwargs...)
    # Value of `t` at which `f` should be called next:
    tnext = Ref(typemax(tType))
    condition = (t, u, integrator) -> t == tnext[]

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        user_affect!(integrator)

        # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
        tnew = time_choice(integrator)
        tstops = integrator.opts.tstops
        for i in length(tstops) : -1 : 1 # reverse iterate to encounter large elements earlier
            if DataStructures.compare(tstops.comparer, tnew, tstops.valtree[i]) # TODO: relying on implementation details
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            end
        end
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_iterative = function (c, t, u, integrator)
        initialize(c, t, u, integrator)
        if initial_affect
            tnext[] = t
            affect!(integrator)
        else
            tnext[] = time_choice(integrator)
            add_tstop!(integrator, tnext[])
        end
    end
    DiscreteCallback(condition, affect!; initialize = initialize_iterative, kwargs...)
end

export IterativeCallback

function PeriodicCallback(f, Δt::Number; initialize = DiffEqBase.INITIALIZE_DEFAULT,
                                         initial_affect = true, kwargs...)
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
            if DataStructures.compare(tstops.comparer, tnew, tstops.valtree[i]) # TODO: relying on implementation details
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            end
        end
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, t, u, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, t, u, integrator)
        if initial_affect
            tnext[] = t
            affect!(integrator)
        else
            tnext[] = time_choice(integrator)
            add_tstop!(integrator, tnext[])
        end
    end

    DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

export PeriodicCallback
