using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5, DiffEqCallbacks
using QuadGK
using Test

include("integrating_GK_shared.jl")

prob = ODEProblem((u, p, t) -> [1.0], [0.0], (0.0, 1.0))
integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(
    prob, Euler(),
    callback = IntegratingGKCallback(
        (u, t, integrator) -> [1.0], integrated, Float64[0.0]
    ),
    dt = 0.1
)
@test all(integrated.integrand .≈ [[0.1] for i in 1:10])

# test to enter recursion
integrated = IntegrandValues(Float64, Vector{Float64})
sol = solve(
    prob, Euler(),
    callback = IntegratingGKCallback(
        (u, t, integrator) -> [cos.(1000 * u[1])], integrated, Float64[0.0]
    ),
    dt = 0.1
)
@test sum(integrated.integrand)[1] .≈ sin(1000) / 1000

#### TESTING ON LINEAR SYSTEM WITH ANALYTICAL SOLUTION ####

u0 = [1.0, 1.0]     # initial condition
tspan = (0.0, 10.0) # simulation time
p = [1.0, 2.0]      # parameters
prob = ODEProblem(simple_linear_system, u0, tspan, p)
sol = solve(prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)

integrand_values = IntegrandValues(Float64, Vector{Float64})
integrand_values_inplace = IntegrandValues(Float64, Vector{Float64})
cb = IntegratingGKCallback(
    (u, t, integrator) -> callback_saving_linear(u, t, integrator, sol),
    integrand_values, zeros(length(p))
)
cb_inplace = IntegratingGKCallback(
    (du, u, t, integrator) -> callback_saving_linear_inplace(
        du,
        u,
        t,
        integrator,
        sol
    ),
    integrand_values_inplace, zeros(length(p))
)
prob_adjoint = ODEProblem(
    (u, p, t) -> adjoint_linear(u, p, t, sol),
    [0.0, 0.0],
    (tspan[end], tspan[1]),
    p,
    callback = cb
)
prob_adjoint_inplace = ODEProblem(
    (du, u, p, t) -> adjoint_linear_inplace(du, u, p, t, sol),
    [0.0, 0.0],
    (tspan[end], tspan[1]),
    p,
    callback = cb_inplace
)
sol_adjoint = solve(prob_adjoint, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
# out-of-place is done
sol_adjoint_inplace = solve(prob_adjoint_inplace, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
# in-place is done

dGdp_new = compute_dGdp(integrand_values)
dGdp_new_inplace = compute_dGdp(integrand_values_inplace)
dGdp_analytical = analytical_derivative(p, tspan[end])

@test isapprox(dGdp_analytical, dGdp_new, atol = 1.0e-11, rtol = 1.0e-11)
@test isapprox(dGdp_analytical, dGdp_new_inplace, atol = 1.0e-11, rtol = 1.0e-11)
