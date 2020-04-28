function IterativeCallback(time_choice, user_affect!,tType = Float64;
                           initialize = DiffEqBase.INITIALIZE_DEFAULT,
                           initial_affect = false, kwargs...)
    # Value of `t` at which `f` should be called next:
    tnext = Ref{Union{Nothing,eltype(tType)}}(typemax(tType))
    condition = function (u, t, integrator)
      t == tnext[]
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        user_affect!(integrator)

        # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
        tnew = time_choice(integrator)
        tnew === nothing && (tnext[] = tnew; return)
        tstops = integrator.opts.tstops
        for i in length(tstops) : -1 : 1 # reverse iterate to encounter large elements earlier
            #=
            Okay yeah, this is nasty
            the comparer is always less than for type stability, so in order
            for this to actually check the correct direction we multiply by
            tdir
            =#
            if DataStructures.compare(tstops.comparer, integrator.tdir*tnew, integrator.tdir*tstops.valtree[i]) # TODO: relying on implementation details
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            elseif tstops.valtree[i] == tnew
              # If it's already a tstop, no need to re-add! This is for the final point
              tnext[] = tnew
            end
        end
        nothing
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_iterative = function (c, u, t, integrator)
        initialize(c, u, t, integrator)
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
    t0 = Ref(typemax(Δt))
    index = Ref(0)
    condition = (u, t, integrator) -> t == (t0[] + index[] * Δt)

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        f(integrator)

        # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
        tnew = t0[] + (index[] + 1) * Δt
        tstops = integrator.opts.tstops
        for i in length(tstops) : -1 : 1 # reverse iterate to encounter large elements earlier
            #=
            Okay yeah, this is nasty
            the comparer is always less than for type stability, so in order
            for this to actually check the correct direction we multiply by
            tdir
            =#
            if DataStructures.compare(tstops.comparer, integrator.tdir*tnew, integrator.tdir*tstops.valtree[i]) # TODO: relying on implementation details
                index[] += 1
                add_tstop!(integrator, tnew)
                break
            end
        end
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, u, t, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, u, t, integrator)
        t0[] = t
        if initial_affect
            index[] = 0
            affect!(integrator)
        else
            index[] = 1
            add_tstop!(integrator, t0[] + Δt)
        end
    end

    DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

export PeriodicCallback
