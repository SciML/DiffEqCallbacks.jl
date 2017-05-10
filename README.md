# DiffEqCallbacks.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqCallbacks.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqCallbacks.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/a3o1a4l4xqcwuw86?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/diffeqcallbacks-jl-ufx45)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DiffEqCallbacks.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DiffEqCallbacks.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/DiffEqCallbacks.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DiffEqCallbacks.jl?branch=master)

[![DiffEqCallbacks](http://pkg.julialang.org/badges/DiffEqCallbacks_0.5.svg)](http://pkg.julialang.org/?pkg=DiffEqCallbacks)
[![DiffEqCallbacks](http://pkg.julialang.org/badges/DiffEqCallbacks_0.6.svg)](http://pkg.julialang.org/?pkg=DiffEqCallbacks)

This is a library of callbacks for extending the solvers of DifferentialEquations.jl.

Currently only one callback is implemented.

## Usage


To use the callbacks provided in this library with DifferentialEquations.jl solvers,
just pass it to the solver via the `callback` keyword argument:

```julia
sol = solve(prob,alg;callback=cb)
```

For more information on using callbacks, [see the manual page](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

## ManifoldProjection

This projects the solution to a manifold, conserving a property while
conserving the order.

```julia
ManifoldProjection(g;nlsolve=NLSOLVEJL_SETUP(),save=true)
```

- `g`: The residual function for the manifold: `g(u,resid)`. This is an inplace function
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
