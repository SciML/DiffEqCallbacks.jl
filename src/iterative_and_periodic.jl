"""
```julia
IterativeCallback(time_choice, user_affect!, tType = Float64;
    initial_affect = false, kwargs...)
```

A callback to be used to iteratively apply some affect. For example, if given the first
effect at `t₁`, you can define `t₂` to apply the next effect.

## Arguments

  - `time_choice(integrator)` determines the time of the next callback. If `nothing` is
    returned for the time choice, then the iterator ends.
  - `user_affect!` is the effect applied to the integrator at the stopping points.

## Keyword Arguments

  - `initial_affect` is whether to apply the affect at `t=0` which defaults to `false`
"""
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
        tstops = get_tstops(integrator)
        #=
        Okay yeah, this is nasty
        the comparer is always less than for type stability, so in order
        for this to actually check the correct direction we multiply by
        tdir
        =#
        tdir_tnew = integrator.tdir * tnew
        tstops_array = get_tstops_array(integrator)
        for i in length(tstops):-1:1 # reverse iterate to encounter large elements earlier
            if tdir_tnew < tstops_array[i] # TODO: relying on implementation details
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            elseif tdir_tnew == tstops_array[i]
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
    add_next_tstop!(integrator, S)

    S.affect!(integrator)
end

function add_next_tstop!(integrator, S)
    @unpack Δt, t0, index = S

    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
    tnew = t0[] + (index[] + 1) * Δt
    #=
    Okay yeah, this is nasty
    the comparer is always less than for type stability, so in order
    for this to actually check the correct direction we multiply by
    tdir
    =#
    tdir_tnew = integrator.tdir * tnew
    index[] += 1
    if tdir_tnew < get_tstops_max(integrator)
        add_tstop!(integrator, tnew)
    end
end

"""
```julia
PeriodicCallback(f, Δt::Number; phase = 0, initial_affect = false,
    final_affect = false,
    kwargs...)
```

`PeriodicCallback` can be used when a function should be called periodically in terms of
integration time (as opposed to wall time), i.e. at `t = tspan[1]`, `t = tspan[1] + Δt`,
`t = tspan[1] + 2Δt`, and so on.

If a non-zero `phase` is provided, the invocations of the callback will be shifted by
`phase` time units, i.e., the calls will occur at
`t = tspan[1] + phase`, `t = tspan[1] + phase + Δt`,
`t = tspan[1] + phase + 2Δt`, and so on.

This callback can, for example, be used to model a
discrete-time controller for a continuous-time system, running at a fixed rate.

## Arguments

  - `f` the `affect!(integrator)` function to be called periodically
  - `Δt` is the period

  ## Keyword Arguments

  - `phase` is a phase offset
  - `initial_affect` is whether to apply the affect at the initial time, which defaults to `false`
  - `final_affect` is whether to apply the affect at the final time, which defaults to `false`
  - `kwargs` are keyword arguments accepted by the `DiscreteCallback` constructor.
"""
function PeriodicCallback(f, Δt::Number;
        phase = 0,
        initial_affect = false,
        final_affect = false,
        initialize = (cb, u, t, integrator) -> u_modified!(integrator,
            initial_affect),
        kwargs...)
    phase < 0 && throw(ArgumentError("phase offset must be non-negative"))
    # Value of `t` at which `f` should be called next:
    t0 = Ref(typemax(Δt))
    index = Ref(0)

    condition = function (u, t, integrator)
        fin = isfinished(integrator)
        (t == (t0[] + index[] * Δt) && !fin) || (final_affect && fin)
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = PeriodicCallbackAffect(f, Δt, t0, index)

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, u, t, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, u, t, integrator)
        t0[] = t + phase
        index[] = iszero(phase) ? 0 : -1
        if initial_affect
            affect!(integrator)
        else
            add_next_tstop!(integrator, affect!)
        end
    end

    DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
           (hasfield(typeof(integrator), :iter) && (integrator.iter == integrator.opts.maxiters))
end

export PeriodicCallback
