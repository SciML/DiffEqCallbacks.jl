# Manifold Projection

The following callbacks are designed to provide post-step modifications to preserve
geometric behaviors in the solution.

```@docs
ManifoldProjection
```

### Example

Here we solve the harmonic oscillator:

```@example manifold
using OrdinaryDiffEq, DiffEqCallbacks, NonlinearSolve, Plots, ADTypes

u0 = ones(2)
function f(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end
prob = ODEProblem(f, u0, (0.0, 100.0))
```

!!! note
    
    Note that NonlinearSolve.jl is required to be imported for ManifoldProjection

However, this problem is supposed to conserve energy, and thus we define our manifold
to conserve the sum of squares:

```@example manifold
function g(resid, u, p, t)
    resid[1] = u[2]^2 + u[1]^2 - 2
end
```

To build the callback, we just call

```@example manifold
cb = ManifoldProjection(g; autodiff = AutoForwardDiff(), resid_prototype = zeros(1))
```

Using this callback, the Runge-Kutta method `Vern7` conserves energy. Note that the
standard saving occurs after the step and before the callback, and thus we set
`save_everystep=false` to turn off all standard saving and let the callback
save after the projection is applied.

```@example manifold
sol = solve(prob, Vern7(), save_everystep = false, callback = cb)
@show sol[end][1]^2 + sol[end][2]^2 ≈ 2
```

```@example manifold
using Plots
plot(sol, idxs = (1, 2))
```
