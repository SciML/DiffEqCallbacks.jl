using OrdinaryDiffEqTsit5, Test, DiffEqBase, DiffEqCallbacks
include("test_problems.jl")

prob = prob_ode_linear

time_choice(integrator) = rand() + integrator.t
affect!(integrator) = integrator.u *= 2
cb = IterativeCallback(time_choice, affect!)

sol = solve(prob, Tsit5(), callback = cb)
