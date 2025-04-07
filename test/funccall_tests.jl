using OrdinaryDiffEqTsit5, DiffEqCallbacks, Test
import ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear

prob = prob_ode_linear
ts = Vector{Float64}()
cb = FunctionCallingCallback((u, t, integrator) -> push!(ts, t))
sol = solve(prob, Tsit5(), callback = cb)
@test !isempty(ts)

ts = Vector{Float64}()
cb = FunctionCallingCallback((u, t, integrator) -> push!(ts, t), funcat = 0.0:0.25:1.0)
sol = solve(prob, Tsit5(), callback = cb)
@test collect(0.0:0.25:1.0) == ts

ts = Vector{Float64}()
cb = FunctionCallingCallback((u, t, integrator) -> push!(ts, t), funcat = 0.25)
sol = solve(prob, Tsit5(), callback = cb)
@test collect(0.0:0.25:1.0) == ts

ts = Vector{Float64}()
cb = FunctionCallingCallback((u, t, integrator) -> push!(ts, t),
    funcat = 0.25,
    func_start = false)
preset = PresetTimeCallback(Float64[], identity)
sol = solve(prob, Tsit5(), callback = CallbackSet(cb, preset))
@test collect(0.25:0.25:1.0) == ts
@test sol.t[2] > 0

ts = Vector{Float64}()
cb = FunctionCallingCallback((u, t, integrator) -> push!(ts, t), funcat = 0.0:0.25:1.0,
    func_everystep = true)
sol = solve(prob, Tsit5(), callback = cb)
@test intersect(collect(0.0:0.25:1.0), ts) == collect(0.0:0.25:1.0)
