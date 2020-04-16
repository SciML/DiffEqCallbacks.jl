# DiffEqCallbacks.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqCallbacks.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqCallbacks.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/a3o1a4l4xqcwuw86?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/diffeqcallbacks-jl-ufx45)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DiffEqCallbacks.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DiffEqCallbacks.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/DiffEqCallbacks.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DiffEqCallbacks.jl?branch=master)

This is a library of callbacks for extending the solvers of DifferentialEquations.jl.

## Usage


To use the callbacks provided in this library with DifferentialEquations.jl solvers,
just pass it to the solver via the `callback` keyword argument:

```julia
sol = solve(prob,alg;callback=cb)
```

For more information on using callbacks, [see the manual page](http://docs.sciml.ai/dev/features/callback_functions).

## ManifoldProjection

This projects the solution to a manifold, conserving a property while
conserving the order.

```julia
ManifoldProjection(g;nlsolve=NLSOLVEJL_SETUP(),save=true)
```

- `g`: The residual function for the manifold: `g(resid,u)`. This is an inplace function
  which writes to the residual the difference from the manifold components.
- `nlsolve`: A nonlinear solver as defined [in the nlsolve format](linear_nonlinear.html)
- `save`: Whether to do the standard saving (applied after the callback)

## AutoAbstol

Many problem solving environments [such as MATLAB](https://www.mathworks.com/help/simulink/gui/absolute-tolerance.html)
provide a way to automatically adapt the absolute tolerance to the problem. This
helps the solvers automatically "learn" what appropriate limits are. Via the
callback interface, DiffEqCallbacks.jl implements a callback `AutoAbstol` which
has the same behavior as the MATLAB implementation, that is the absolute tolerance
starts at `init_curmax` (default `1-e6`), and at each iteration it is set
to the maximum value that the state has thus far reached times the relative tolerance.

To generate the callback, use the constructor:

```julia
AutoAbstol(save=true;init_curmax=1e-6)
```

`save` determines whether this callback has saving enabled, and `init_curmax` is
the initial `abstol`. If this callback is used in isolation, `save=true` is required
for normal saving behavior. Otherwise, `save=false` should be set to ensure
extra saves do not occur.

## Domain Controls

The domain controls are efficient methods for preserving a domain relation for
the solution value `u`. Unlike the `isoutofdomain` method, these methods use
interpolations and extrapolations to more efficiently choose stepsizes, but
require that the solution is well defined slightly outside of the domain.

### PositiveDomain

```julia
PositiveDomain(u=nothing; save=true, abstol=nothing, scalefactor=nothing)
```

### GeneralDomain

```julia
GeneralDomain(g, u=nothing; nlsolve=NLSOLVEJL_SETUP(), save=true,
                       abstol=nothing, scalefactor=nothing, autonomous=numargs(g)==2,
                       nlopts=Dict(:ftol => 10*eps()))
```

## StepsizeLimiter

The stepsize limiter lets you define a function `dtFE(u,p,t)` which changes the
allowed maximal stepsize throughout the computation. The constructor is:

```julia
StepsizeLimiter(dtFE;safety_factor=9//10,max_step=false,cached_dtcache=0.0)
```

`dtFE` is the maximal timestep and is calculated using the previous `t` and `u`.
`safety_factor` is the factor below the true maximum that will be stepped to
which defaults to `9//10`. `max_step=true` makes every step equal to
`safety_factor*dtFE(u,p,t)` when the solver is set to `adaptive=false`. `cached_dtcache`
should be set to match the type for time when not using Float64 values.

## FunctionCallingCallback

The function calling callback lets you define a function `func(u,t,integrator)`
which gets calls at the time points of interest. The constructor is:

```julia
FunctionCallingCallback(func;
               funcat=Vector{Float64}(),
               func_everystep=isempty(funcat),
               func_start = true
               tdir=1)
```
- `func(t, u, integrator)` is the function to be called.
- `funcat` values that the function is sure to be evaluated at.
- `func_everystep` whether to call the function after each integrator step.
- `func_start` whether the function is called the initial condition.
- `tdir` should be `sign(tspan[end]-tspan[1])`. It defaults to `1` and should
  be adapted if `tspan[1] > tspan[end]`.

## SavingCallback

The saving callback lets you define a function `save_func(u, t, integrator)` which
returns quantities of interest that shall be saved. The constructor is:

```julia
SavingCallback(save_func, saved_values::SavedValues;
               saveat=Vector{eltype(saved_values.t)}(),
               save_everystep=isempty(saveat),
               save_start = true,
               tdir=1)
```
- `save_func(u, t, integrator)` returns the quantities which shall be saved.
  Note that this should allocate the output (not as a view to `u`).
- `saved_values::SavedValues` is the types that `save_func` will return, i.e.
  `save_func(t, u, integrator)::savevalType`. It's specified via
  `SavedValues(typeof(t),savevalType)`, i.e. give the type for time and the
  type that `save_func` will output (or higher compatible type).
- `saveat` mimics `saveat` in `solve` from `solve`.
- `save_everystep` mimics `save_everystep` from `solve`.
- `save_start` mimics `save_start` from `solve`.
- `tdir` should be `sign(tspan[end]-tspan[1])`. It defaults to `1` and should
  be adapted if `tspan[1] > tspan[end]`.

The outputted values are saved into `saved_values`. Time points are found via
`saved_values.t` and the values are `saved_values.saveval`.

## PresetTimeCallback

`PresetTimeCallback` is a callback that adds callback `affect!` calls at preset
times. No playing around with `tstops` or anything is required: this callback
adds the triggers for you to make it automatic.

```julia
PresetTimeCallback(tstops,user_affect!;
                            initialize = DiffEqBase.INITIALIZE_DEFAULT,
                            filter_tstops = true,
                            kwargs...)
```

- `tstops`: the times for the `affect!` to trigger at.
- `user_affect!`: an `affect!(integrator)` function to use at the time points.
- `filter_tstops`: Whether to filter out tstops beyond the end of the integration timespan.
  Defaults to true. If false, then tstops can extend the interval of integration.

## IterativeCallback

`IterativeCallback` is a callback to be used to iteratively apply some affect.
For example, if given the first effect at `t₁`, you can define `t₂` to apply
the next effect.

A `IterativeCallback` is constructed as follows:

```julia
function IterativeCallback(time_choice, user_affect!,tType = Float64;
                           initial_affect = false, kwargs...)
```

where `time_choice(integrator)` determines the time of the next callback and
`user_affect!` is the effect applied to the integrator at the stopping points.
If `nothing` is returned for the time choice then the iterator ends. `initial_affect`
is whether to apply the affect at `t=0` which defaults to `false`

## PeriodicCallback

`PeriodicCallback` can be used when a function should be called periodically in terms of integration time (as opposed to wall time), i.e. at `t = tspan[1]`, `t = tspan[1] + Δt`, `t = tspan[1] + 2Δt`, and so on. This callback can, for example, be used to model a discrete-time controller for a continuous-time system, running at a fixed rate.

A `PeriodicCallback` can be constructed as follows:

```julia
PeriodicCallback(f, Δt::Number; initial_affect = true, kwargs...)
```

where `f` is the function to be called periodically, `Δt` is the period, `initial_affect` is whether to apply
the affect at `t=0` which defaults to `true`, and `kwargs` are keyword arguments accepted by the `DiscreteCallback` constructor.

## TerminateSteadyState

`TerminateSteadyState` can be used to solve the problem for the steady-state
by running the solver until the derivatives of the problem converge to 0 or
`tspan[2]` is reached. This is an alternative approach to root finding (see
the [Steady State Solvers](@ref) section). The constructor of this callback is:

```julia
TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass)
```

where `abstol` and `reltol` are the absolute and relative tolerance, respectively.
These tolerances may be specified as scalars or as arrays of the same length
as the states of the problem. `test` represents the function that evaluates the
condition for termination. The default condition is that all derivatives should
become smaller than `abstol` and the states times `reltol`. The user
can pass any other function to implement a different termination condition. Such
function should take four arguments: `integrator` (see [Integrator Interface](@ref)
for details), `abstol` and `reltol`.
