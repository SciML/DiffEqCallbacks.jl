using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5, DiffEqCallbacks
using Test

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