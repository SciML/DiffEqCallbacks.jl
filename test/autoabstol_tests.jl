using OrdinaryDiffEq, DiffEqProblemLibrary, Base.Test, DiffEqBase, DiffEqCallbacks
prob = prob_ode_linear

cb = AutoAbstol()
integrator = init(prob,BS3(),callback=cb)
at1 = integrator.opts.abstol
step!(integrator)
at2 = integrator.opts.abstol
@test at1 < at2
step!(integrator)
at3 = integrator.opts.abstol
@test at2 < at3

solve(prob,BS3(),callback=cb)
