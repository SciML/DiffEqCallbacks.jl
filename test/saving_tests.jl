using Base.Test, OrdinaryDiffEq, DiffEqProblemLibrary, DiffEqCallbacks
using Plots; unicodeplots()

# save_everystep, scalar problem
prob = prob_ode_linear
saved_values = SavedValues(Float64, Float64)
cb = SavingCallback((t,u,integrator)->u, saved_values)
sol = solve(prob, Tsit5(), callback=cb)
print("\n", saved_values, "\n")
plot(saved_values)
@test all(idx -> sol.t[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> sol.u[idx] == saved_values.saveval[idx], eachindex(saved_values.t))

# save_everystep, inplace problem
prob2D = prob_ode_2Dlinear
saved_values = SavedValues(eltype(prob2D.tspan), typeof(prob2D.u0))
cb = SavingCallback((t,u,integrator)->copy(u), saved_values)
sol = solve(prob2D, Tsit5(), callback=cb)
@test all(idx -> sol.t[idx] .== saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> all(sol.u[idx] .== saved_values.saveval[idx]), eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D.tspan), eltype(prob2D.u0))
cb = SavingCallback((t,u,integrator)->u[1], saved_values)
sol = solve(prob2D, Tsit5(), callback=cb)
@test all(idx -> sol.t[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> sol.u[idx][1] == saved_values.saveval[idx], eachindex(saved_values.t))

# saveat, scalar problem
saved_values = SavedValues(Float64, Float64)
saveat = linspace(prob.tspan..., 10)
cb = SavingCallback((t,u,integrator)->u, saved_values, saveat=saveat)
sol = solve(prob, Tsit5(), callback=cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 5.e-15, eachindex(saved_values.t))

# saveat, inplace problem
saved_values = SavedValues(eltype(prob2D.tspan), typeof(prob2D.u0))
saveat = linspace(prob2D.tspan..., 10)
cb = SavingCallback((t,u,integrator)->copy(u), saved_values, saveat=saveat)
sol = solve(prob2D, Tsit5(), callback=cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> norm(sol(saveat[idx]) - saved_values.saveval[idx]) < 5.e-15, eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D.tspan), eltype(prob2D.u0))
saveat = linspace(prob2D.tspan..., 10)
cb = SavingCallback((t,u,integrator)->u[1], saved_values, saveat=saveat)
sol = solve(prob2D, Tsit5(), callback=cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx])[1] - saved_values.saveval[idx]) < 5.e-15, eachindex(saved_values.t))

# saveat, tdir<0, scalar problem
prob_inverse = ODEProblem(prob.f, prob.u0, (prob.tspan[end], prob.tspan[1]))
saved_values = SavedValues(Float64, Float64)
saveat = linspace(prob_inverse.tspan..., 10)
cb = SavingCallback((t,u,integrator)->u, saved_values, saveat=saveat, tdir=-1)
sol = solve(prob_inverse, Tsit5(), callback=cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 5.e-15, eachindex(saved_values.t))

# saveat, tdir<0, inplace problem
prob2D_inverse = ODEProblem(prob2D.f, prob2D.u0, (prob2D.tspan[end], prob2D.tspan[1]))
saved_values = SavedValues(eltype(prob2D_inverse.tspan), typeof(prob2D_inverse.u0))
saveat = linspace(prob2D_inverse.tspan..., 10)
cb = SavingCallback((t,u,integrator)->copy(u), saved_values, saveat=saveat, tdir=-1)
sol = solve(prob2D_inverse, Tsit5(), callback=cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> norm(sol(saveat[idx]) - saved_values.saveval[idx]) < 5.e-15, eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D_inverse.tspan), eltype(prob2D_inverse.u0))
saveat = linspace(prob2D_inverse.tspan..., 10)
cb = SavingCallback((t,u,integrator)->u[1], saved_values, saveat=saveat, tdir=-1)
sol = solve(prob2D_inverse, Tsit5(), callback=cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx])[1] - saved_values.saveval[idx]) < 5.e-15, eachindex(saved_values.t))
