using DiffEqCallbacks, DiffEqBase, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5
using SciMLBase, Test

function g(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    return du[3] = x * y - β * z
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 10.0)
p = [10.0, 28.0, 8 / 3]
prob = ODEProblem(g, u0, tspan, p)

cb = ProbIntsUncertainty(1.0e4, 5)
solve(prob, Tsit5())
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 10, callback = cb, adaptive = false,
    dt = 1 / 10
)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

function fitz(du, u, p, t)
    V, R = u
    du[1] = 3.0 * (V - V^3 / 3 + R)
    return du[2] = -(1 / 3.0) * (V - 0.2 - 0.2 * R)
end
u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(fitz, u0, tspan)

cb = ProbIntsUncertainty(0.1, 1)
sol = solve(prob, Euler(), dt = 1 / 10)
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Euler(), trajectories = 100, callback = cb, adaptive = false,
    dt = 1 / 10
)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

cb = AdaptiveProbIntsUncertainty(5)
sol = solve(prob, Tsit5())
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 100, callback = cb, abstol = 1.0e-3,
    reltol = 1.0e-1
)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

# ProbInts must fill the integrator tmp cache with randn! instead of allocating
# `randn(size(u))` on every accepted step.
function _osc!(du, u, p, t)
    du[1] = -0.001 * u[1] + u[2]
    du[2] = -u[1] - 0.001 * u[2]
    du[3] = -0.001 * u[3] + 2.0 * u[4]
    du[4] = -2.0 * u[3] - 0.001 * u[4]
    return nothing
end
function _probints_bytes_per_step(n)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        _osc!, [1.0, 0.0, 0.5, 0.0], (0.0, 1.0e4)
    )
    integ = init(
        prob, Tsit5(); callback = ProbIntsUncertainty(0.01, 5), save_everystep = false
    )
    for _ in 1:20
        step!(integ)
    end
    step!(integ)
    return (
        @allocated for _ in 1:n
            step!(integ)
        end
    ) / n
end
@test _probints_bytes_per_step(200) < 200.0
