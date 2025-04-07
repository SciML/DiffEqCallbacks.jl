using DiffEqCallbacks, DiffEqBase, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5
using Test

function g(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 10.0)
p = [10.0, 28.0, 8 / 3]
prob = ODEProblem(g, u0, tspan, p)

cb = ProbIntsUncertainty(1e4, 5)
solve(prob, Tsit5())
monte_prob = EnsembleProblem(prob)
sim = solve(monte_prob, Tsit5(), trajectories = 10, callback = cb, adaptive = false,
    dt = 1 / 10)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

function fitz(du, u, p, t)
    V, R = u
    du[1] = 3.0 * (V - V^3 / 3 + R)
    du[2] = -(1 / 3.0) * (V - 0.2 - 0.2 * R)
end
u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(fitz, u0, tspan)

cb = ProbIntsUncertainty(0.1, 1)
sol = solve(prob, Euler(), dt = 1 / 10)
monte_prob = EnsembleProblem(prob)
sim = solve(monte_prob, Euler(), trajectories = 100, callback = cb, adaptive = false,
    dt = 1 / 10)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

cb = AdaptiveProbIntsUncertainty(5)
sol = solve(prob, Tsit5())
monte_prob = EnsembleProblem(prob)
sim = solve(monte_prob, Tsit5(), trajectories = 100, callback = cb, abstol = 1e-3,
    reltol = 1e-1)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)
