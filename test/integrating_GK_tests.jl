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

# enters reccursion
integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(prob, Euler(),
    callback = IntegratingGKCallback(
        (u, t, integrator) -> [cos.(1000*u[1])], integrated, Float64[0.0]),
    dt = 0.1)
@test sum(integrated.integrand)[1] .≈ sin(1000)/1000
