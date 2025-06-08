using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5, SciMLSensitivity, DiffEqCallbacks,
      Zygote
using ForwardDiff
using QuadGK
using Test

prob = ODEProblem((u, p, t) -> [1.0], [0.0], (0.0, 1.0))
integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(prob, Euler(),
    callback = IntegratingGKCallback(
        (u, t, integrator) -> [1.0], integrated, Float64[0.0]),
    dt = 0.1)
@test all(integrated.integrand .≈ [[0.1] for i in 1:10])
