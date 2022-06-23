using OrdinaryDiffEq, DiffEqProblemLibrary, Test, DiffEqBase, DiffEqCallbacks

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear


prob = prob_ode_linear

time_choice(integrator) = rand() + integrator.t
affect!(integrator) = integrator.u *= 2
cb = IterativeCallback(time_choice, affect!)

sol = solve(prob, Tsit5(), callback = cb)
