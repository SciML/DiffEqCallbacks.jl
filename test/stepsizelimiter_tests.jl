using OrdinaryDiffEq, DiffEqCallbacks, Test

f2(u, p, t) = 1.01 * u
u0 = 1 / 2
tspan = (0.0, 5.0)
prob = ODEProblem(f2, u0, tspan)

sol = solve(prob, Tsit5())
@test maximum(diff(sol.t)) > 0.46

dtFE(u, p, t) = 1 / 20
cb = StepsizeLimiter(dtFE)
sol = solve(prob, Tsit5(), callback = cb)
@test maximum(diff(sol.t)) < 0.046001

cb = StepsizeLimiter(dtFE; safety_factor = 1)
sol = solve(prob, Tsit5(), callback = cb)
@test maximum(diff(sol.t)) < 0.050001

cb = StepsizeLimiter(dtFE; safety_factor = 1)
sol = solve(prob, Tsit5(), callback = cb, adaptive = false, dt = 1 / 10)
@test maximum(diff(sol.t)) < 0.050001

dtFE(u, p, t) = t < 1 ? 1 / 20 : 1 / 10
cb = StepsizeLimiter(dtFE; safety_factor = 1)
sol = solve(prob, Tsit5(), callback = cb, adaptive = false, dt = 1 / 10)
@test maximum(diff(sol.t)) > 0.050001
@test maximum(diff(sol.t)) < 0.100001

sol = solve(prob, Tsit5(), callback = cb, adaptive = false, dt = 1 / 20)
@test maximum(diff(sol.t)) < 0.050001

dtFE(u, p, t) = t + 0.2
cb = StepsizeLimiter(dtFE; safety_factor = 1, max_step = true)
sol = solve(prob, Tsit5(), callback = cb, adaptive = false, dt = 1 / 20)
@test diff(sol.t) ≈ [0.2, 0.4, 0.8, 1.6, 2.0]
