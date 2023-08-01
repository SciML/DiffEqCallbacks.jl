# Output and Saving Controls

These callbacks extend the output and saving controls available during time stepping.

```@docs
SavingCallback
FunctionCallingCallback
```

### Saving Example

In this example, we will solve a matrix equation and at each step save a tuple
of values which contains the current trace and the norm of the matrix. We build
the `SavedValues` cache to use `Float64` for time and `Tuple{Float64,Float64}`
for the saved values, and then call the solver with the callback.

```@example saving
using DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra
prob = ODEProblem((du, u, p, t) -> du .= u, rand(4, 4), (0.0, 1.0))
saved_values = SavedValues(Float64, Tuple{Float64, Float64})
cb = SavingCallback((u, t, integrator) -> (tr(u), norm(u)), saved_values)
sol = solve(prob, Tsit5(), callback = cb)

print(saved_values.saveval)
```

Note that the values are retrieved from the cache as `.saveval`, and the time points are found as
`.t`. If we want to control the saved times, we use `saveat` in the callback. The save controls like
`saveat` act analogously to how they act in the `solve` function.

```@example saving
saved_values = SavedValues(Float64, Tuple{Float64, Float64})
cb = SavingCallback((u, t, integrator) -> (tr(u), norm(u)), saved_values,
    saveat = 0.0:0.1:1.0)
sol = solve(prob, Tsit5(), callback = cb)
print(saved_values.saveval)
print(saved_values.t)
```
