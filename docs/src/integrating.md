# Numerical Integration Callbacks

Sometimes one may want to solve an integral simultaniously to the solution of a differential equation. For example,
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

```@docs
IntegratingCallback
```
