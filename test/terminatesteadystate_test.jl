using OrdinaryDiffEq, Base.Test, DiffEqBase, DiffEqCallbacks

model(u,t,p) = -0.1.*u
u0 = [1.0, 10.0]
tspan = (0.0, 300.0)
prob = ODEProblem(model, u0, tspan)
sim = solve(prob, Tsit5(), callback=TerminateSteadyState())

@test all(sim(sim.t[end],Val{1}) .< 1e-8)
@test sim.t[end] < tspan[2]
