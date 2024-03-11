using OrdinaryDiffEq, Test, DiffEqBase, DiffEqCallbacks
import ODEProblemLibrary: prob_ode_linear

prob = prob_ode_linear

time_choice(integrator) = rand() + integrator.t
affect!(integrator) = integrator.u *= 2
cb = IterativeCallback(time_choice, affect!)

sol = solve(prob, Tsit5(), callback = cb)

# Fix indexing repeats
# https://github.com/SciML/ModelingToolkit.jl/issues/2528

function lineardecay(du, u, p, t)
    du[1] = -u[1]
end

function bumpaffect!(integ)
    integ.u[1] += 10
end

cb = PeriodicCallback(bumpaffect!, 24.0)
prob = ODEProblem(lineardecay, [0.0], (0.0, 130.0))
sol1 = solve(prob, Tsit5(), callback = cb)

@test sol1(0.0) == [0.0]
@test sol1(24.0 + eps(24.0)) ≈ [10.0]
@test sol1(48.0 + eps(48.0)) ≈ [10.0]
@test sol1(72.0 + eps(72.0)) ≈ [10.0]
@test sol1(96.0 + eps(96.0)) ≈ [10.0]
@test sol1(120.0 + eps(120.0)) ≈ [10.0]
sol2 = solve(prob, Tsit5(), callback = cb)
@test sol2(0.0) == [0.0]
@test sol2(24.0 + eps(24.0)) ≈ [10.0]
@test sol2(48.0 + eps(48.0)) ≈ [10.0]
@test sol2(72.0 + eps(72.0)) ≈ [10.0]
@test sol2(96.0 + eps(96.0)) ≈ [10.0]
@test sol2(120.0 + eps(120.0)) ≈ [10.0]
