using OrdinaryDiffEqTsit5, Test, DiffEqBase, DiffEqCallbacks
import ODEProblemLibrary: prob_ode_linear

prob = prob_ode_linear

time_choice(integrator) = rand() + integrator.t
iterative_affect!(integrator) = integrator.u *= 2
cb = IterativeCallback(time_choice, iterative_affect!)

sol = solve(prob, Tsit5(), callback = cb)
