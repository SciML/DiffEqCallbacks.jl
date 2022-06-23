
function IterativeCallback(time_choice, user_affect!, tType = Float64;
                           initial_affect = false,
                           initialize = (cb, u, t, integrator) -> u_modified!(integrator,
                                                                              initial_affect),
                           kwargs...)
    # Value of `t` at which `f` should be called next:
    tnext = Ref{Union{Nothing, eltype(tType)}}(typemax(tType))
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
        #=
        Okay yeah, this is nasty
        the comparer is always less than for type stability, so in order
        for this to actually check the correct direction we multiply by
        tdir
        =#
        tdir_tnew = integrator.tdir * tnew
        for i in length(tstops):-1:1 # reverse iterate to encounter large elements earlier
            if tdir_tnew < tstops.valtree[i] # TODO: relying on implementation details
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            elseif tdir_tnew == tstops.valtree[i]
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
            if tnext[] != nothing
                add_tstop!(integrator, tnext[])
            end
        end
    end
    DiscreteCallback(condition, affect!; initialize = initialize_iterative, kwargs...)
end

export IterativeCallback

struct PeriodicCallbackAffect{A, dT, Ref1, Ref2}
    affect!::A
    Δt::dT
    t0::Ref1
    index::Ref2
end

function (S::PeriodicCallbackAffect)(integrator)
    @unpack affect!, Δt, t0, index = S

    affect!(integrator)

    tstops = integrator.opts.tstops

    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
    tnew = t0[] + (index[] + 1) * Δt
    tstops = integrator.opts.tstops
    #=
    Okay yeah, this is nasty
    the comparer is always less than for type stability, so in order
    for this to actually check the correct direction we multiply by
    tdir
    =#
    tdir_tnew = integrator.tdir * tnew
    for i in length(tstops):-1:1 # reverse iterate to encounter large elements earlier
        if tdir_tnew < tstops.valtree[i] # TODO: relying on implementation details
            index[] += 1
            add_tstop!(integrator, tnew)
            break
        end
    end
end

function PeriodicCallback(f, Δt::Number;
                          initial_affect = false,
                          initialize = (cb, u, t, integrator) -> u_modified!(integrator,
                                                                             initial_affect),
                          kwargs...)

    # Value of `t` at which `f` should be called next:
    t0 = Ref(typemax(Δt))
    index = Ref(0)
    condition = (u, t, integrator) -> t == (t0[] + index[] * Δt)

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = PeriodicCallbackAffect(f, Δt, t0, index)

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
