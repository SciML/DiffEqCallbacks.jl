using OrdinaryDiffEq, Base.Test, DiffEqBase, DiffEqCallbacks, StaticArrays

# In-place modification of du (as array)
model(du,u,t,p) = du .= -0.1.*u
u0 = [1.0, 10.0]
tspan = (0.0, 1000.0)
prob = ODEProblem(model, u0, tspan)
sim = solve(prob, Tsit5(), callback=TerminateSteadyState())

@test all(sim(sim.t[end],Val{1}) .< 1e-8)
@test sim.t[end] < tspan[2]

# Out-of-place modification of du (as array)
smodel(u,t,p) = -0.1.*u
su0 = @SVector [1.0, 10.0]
tspan = (0.0, 1000.0)
prob = ODEProblem(smodel, su0, tspan)
sim = solve(prob, Tsit5(), callback=TerminateSteadyState())

@test all(sim(sim.t[end],Val{1}) .< 1e-8)
@test sim.t[end] < tspan[2]
