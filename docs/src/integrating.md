# Numerical Integration Callbacks

Sometimes one may want to solve an integral simultaneously to the solution of a differential equation. For example,
assume we want to solve:

```math
u^\prime = f(u,p,t)
h = \int_{t_0}^{t_f} g(u,p,t) dt
```

While one can use the ODE solver's dense solution to call an integration scheme on `sol(t)` after the solve, this can
be memory intensive. Another way one can solve this problem is by extending the system, i.e.:

```math
u^\prime = f(u,p,t)
h^\prime = g(u,p,t)
```

with $h(t_0) = 0$, so then $h(t_f)$ would be the solution to the integral. However, many differential equation solvers
scale superlinearly with the equation size and thus this could add an extra cost to the solver process.

The `IntegratingCallback` allows one to be able to solve such definite integrals in a way that is both memory and compute
efficient. It uses the free local interpolation of a given step in order to approximate the Gaussian quadrature for a given
step to the order of the numerical differential equation solve, thus achieving accuracy while not requiring the post-solution
dense interpolation to be saved. By doing this via a callback, this method is able to easily integrate with functionality
that introduces discontinuities, like other callbacks, in a way that is more accurate than a direct integration post solve.

The `IntegratingSumCallback` is the same, but instead of returning the timeseries of the interval results of the integration,
it simply returns the final integral value.

The `IntegratingGKCallback` uses Gauss-Kronrod quadrature method in order to allow for error control.

```@docs
IntegratingCallback
IntegrandValues
IntegratingSumCallback
IntegrandValuesSum
IntegratingGKCallback
```

## Example

```@example integrating
using OrdinaryDiffEq, DiffEqCallbacks, Test
prob = ODEProblem((u, p, t) -> [1.0], [0.0], (0.0, 1.0))
integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(prob, Euler(),
    callback = IntegratingCallback(
        (u, t, integrator) -> [1.0], integrated, Float64[0.0]),
    dt = 0.1)
@test all(integrated.integrand .≈ [[0.1] for i in 1:10])

integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(prob, Euler(),
    callback = IntegratingCallback(
        (u, t, integrator) -> [u[1]], integrated, Float64[0.0]),
    dt = 0.1)
@test all(integrated.integrand .≈ [[((n * 0.1)^2 - ((n - 1) * (0.1))^2) / 2] for n in 1:10])
@test sum(integrated.integrand)[1] ≈ 0.5

integrated = IntegrandValuesSum(zeros(1))
sol = solve(prob, Euler(),
    callback = IntegratingSumCallback(
        (u, t, integrator) -> [1.0], integrated, Float64[0.0]),
    dt = 0.1)
@test integrated.integrand[1] == 1
integrated = IntegrandValuesSum(zeros(1))
sol = solve(prob, Euler(),
    callback = IntegratingSumCallback(
        (u, t, integrator) -> [u[1]], integrated, Float64[0.0]),
    dt = 0.1)
@test integrated.integrand[1] == 0.5

integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(prob, Euler(),
    callback = IntegratingGKCallback(
        (u, t, integrator) -> [cos.(1000*u[1])], integrated, Float64[0.0], 1e-7),
    dt = 0.1)
@test sum(integrated.integrand)[1] .≈ sin(1000)/1000
```
