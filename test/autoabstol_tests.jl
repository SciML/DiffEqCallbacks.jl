using OrdinaryDiffEq, Test, DiffEqBase, DiffEqCallbacks
import ODEProblemLibrary: prob_ode_2Dlinear, prob_ode_linear

prob = prob_ode_linear

cb = AutoAbstol()
integrator = init(prob, BS3(), callback = cb)
at1 = integrator.opts.abstol
step!(integrator)
at2 = integrator.opts.abstol
@test at1 < at2
step!(integrator)
at3 = integrator.opts.abstol
@test at2 < at3

solve(prob, BS3(), callback = cb)

prob2D = prob_ode_2Dlinear
cb2D = AutoAbstol(init_curmax = zeros(4, 2))
integrator2D_1 = init(prob2D, BS3(), callback = cb, abstol = 1e-6)
integrator2D_2 = init(prob2D, BS3(), callback = cb, abstol = fill(1e-6, 4, 2))
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)
step!(integrator2D_1)
step!(integrator2D_2)
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)
step!(integrator2D_1)
step!(integrator2D_2)
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)

sol1 = solve(prob2D, BS3(), callback = cb2D, abstol = fill(1e-6, 4, 2))
sol2 = solve(prob2D, BS3(), callback = cb2D, abstol = fill(1e-6, 4, 2),
    reltol = fill(1e-3, 4, 2))
@test sol1.t == sol2.t && sol1.u == sol2.u
@test_throws MethodError solve(prob, BS3(), callback = cb2D, abstol = 1e-6)
