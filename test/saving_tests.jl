using Test, OrdinaryDiffEq, DiffEqCallbacks, LinearAlgebra,
      SciMLSensitivity, Tracker
import LinearAlgebra: norm
import ODEProblemLibrary: prob_ode_2Dlinear, prob_ode_linear

# save_everystep, scalar problem
prob = prob_ode_linear
saved_values = SavedValues(Float64, Float64)
cb = SavingCallback((u, t, integrator) -> u, saved_values)
sol = solve(prob, Tsit5(), callback = cb)
print("\n", saved_values, "\n")
@test all(idx -> sol.t[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> sol.u[idx] == saved_values.saveval[idx], eachindex(saved_values.t))

# save_everystep, inplace problem
prob2D = prob_ode_2Dlinear
saved_values = SavedValues(eltype(prob2D.tspan), typeof(prob2D.u0))
cb = SavingCallback((u, t, integrator) -> copy(u), saved_values)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> sol.t[idx] .== saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> all(sol.u[idx] .== saved_values.saveval[idx]), eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D.tspan), eltype(prob2D.u0))
cb = SavingCallback((u, t, integrator) -> u[1], saved_values)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> sol.t[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> sol.u[idx][1] == saved_values.saveval[idx], eachindex(saved_values.t))

# saveat, scalar problem
saved_values = SavedValues(Float64, Float64)
saveat = range(prob.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u, saved_values, saveat = saveat)
sol = solve(prob, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
          eachindex(saved_values.t))

# saveat, inplace problem
saved_values = SavedValues(eltype(prob2D.tspan), typeof(prob2D.u0))
saveat = range(prob2D.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> copy(u), saved_values, saveat = saveat)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> norm(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
          eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D.tspan), eltype(prob2D.u0))
saveat = range(prob2D.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u[1], saved_values, saveat = saveat)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx])[1] - saved_values.saveval[idx]) < 8.e-15,
          eachindex(saved_values.t))

# saveat, tdir<0, scalar problem
prob_inverse = ODEProblem(prob.f, prob.u0, (prob.tspan[end], prob.tspan[1]), 1.01)
saved_values = SavedValues(Float64, Float64)
saveat = range(prob_inverse.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u, saved_values, saveat = saveat, tdir = -1)
sol = solve(prob_inverse, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
          eachindex(saved_values.t))

# saveat, tdir<0, inplace problem
prob2D_inverse = ODEProblem(prob2D.f, prob2D.u0, (prob2D.tspan[end], prob2D.tspan[1]), 1.01)
saved_values = SavedValues(eltype(prob2D_inverse.tspan), typeof(prob2D_inverse.u0))
saveat = range(prob2D_inverse.tspan[1], stop = prob2D_inverse.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> copy(u), saved_values, saveat = saveat, tdir = -1)
sol = solve(prob2D_inverse, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> norm(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
          eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D_inverse.tspan), eltype(prob2D_inverse.u0))
saveat = range(prob2D_inverse.tspan[1], stop = prob2D_inverse.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u[1], saved_values, saveat = saveat, tdir = -1)
sol = solve(prob2D_inverse, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx])[1] - saved_values.saveval[idx]) < 8.e-15,
          eachindex(saved_values.t))

# Make sure it doesn't error with mutable in oop
prob = ODEProblem((u, p, t) -> u, rand(4, 4), (0.0, 1.0))
saved_values = SavedValues(Float64, Tuple{Float64, Float64})
cb = SavingCallback((u, t, integrator) -> (tr(u), norm(u)), saved_values,
                    saveat = 0.0:0.1:1.0)
sol = solve(prob, Tsit5(), callback = cb)
println(saved_values.saveval)

# Save only end
prob = ODEProblem((du, u, p, t) -> du .= u, rand(4, 4), (0.0, 1.0))
saved_values = SavedValues(Float64, Tuple{Float64, Float64})
cb = SavingCallback((u, t, integrator) -> (tr(u), norm(u)), saved_values,
                    save_everystep = false, save_start = false)
sol = solve(prob, Tsit5(), callback = cb)
print(saved_values.saveval)
@test length(saved_values.t) == 1
@test saved_values.t[1] == 1.0

# Tracker with Saving Callback
## This is pretty much a hack. It has been merged into DistributionsAD master
Base.prevfloat(r::Tracker.TrackedReal) = Tracker.track(prevfloat, r)
Tracker.@grad function prevfloat(r::Real)
    prevfloat(Tracker.data(r)), Δ -> (Δ,)
end
Base.nextfloat(r::Tracker.TrackedReal) = Tracker.track(nextfloat, r)
Tracker.@grad function nextfloat(r::Real)
    nextfloat(Tracker.data(r)), Δ -> (Δ,)
end

function rober(u, p::TrackedArray, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return Tracker.collect([-k₁ * y₁ + k₃ * y₂ * y₃,
                               k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃,
                               k₂ * y₂^2])
end

p = TrackedArray([1.9f0, 1.0f0, 3.0f0])
u0 = TrackedArray([1.0f0, 0.0f0, 0.0f0])
tspan = TrackedArray([0.0f0, 1.0f0])
prob = ODEProblem{false}(rober, u0, tspan, p)
saved_values = SavedValues(eltype(tspan), eltype(p))
cb = SavingCallback((u, t, integrator) -> integrator.EEst * integrator.dt, saved_values)

@test !all(iszero.(Tracker.gradient(p -> begin
                                        solve(remake(prob, u0 = u0, p = p, tspan = tspan),
                                              Tsit5(),
                                              sensealg = SensitivityADPassThrough(),
                                              callback = cb)
                                        return sum(saved_values.saveval)
                                    end,
                                    p)[1]))
