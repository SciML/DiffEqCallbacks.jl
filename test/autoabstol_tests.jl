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

prob2D = prob_ode_2Dlinear
cb2D = AutoAbstol(init_curmax=zeros(2))
integrator2D_1 = init(prob,BS3(),callback=cb,abstol=1e-6)
integrator2D_2 = init(prob,BS3(),callback=cb,abstol=[1e-6,1e-6])
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)
step!(integrator2D_1)
step!(integrator2D_2)
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)
step!(integrator2D_1)
step!(integrator2D_2)
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)

sol1 = solve(prob,BS3(),callback=cb2D,abstol=[1e-6,1e-6])
sol2 = solve(prob,BS3(),callback=cb2D,abstol=[1e-6,1e-6],reltol=[1e-3,1e-3])
@test sol1.t == sol2.t && sol1.u == sol2.u
@test_throws MethodError solve(prob,BS3(),callback=cb2D,abstol=1e-6)
