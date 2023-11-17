using OrdinaryDiffEq, Test, DiffEqBase, DiffEqCallbacks, StaticArrays

# In-place modification of du (as array)
model(du, u, t, p) = du .= -0.1 .* u
u0 = [1.0, 10.0]
tspan = (0.0, 5000.0)
prob = ODEProblem(model, u0, tspan)
sim = solve(prob, Tsit5(), callback = TerminateSteadyState(1e-6))

@test all(sim(sim.t[end], Val{1}) .< 1e-8)
@test sim.t[end] < tspan[2]

sim2 = solve(prob, Tsit5(), callback = TerminateSteadyState(1e-6, min_t = 900))

@test all(sim2(sim2.t[end], Val{1}) .< 1e-8)
@test sim2.t[end] < tspan[2]
@test sim2.t[end] > sim.t[end]
@test sim2.t[end] > 900

# Out-of-place modification of du (as array)
smodel(u, t, p) = -0.1 .* u
su0 = @SVector [1.0, 10.0]
tspan = (0.0, 1000.0)
prob = ODEProblem(smodel, su0, tspan)
sim = solve(prob, Tsit5(), callback = TerminateSteadyState())

@test all(sim(sim.t[end], Val{1}) .< 1e-8)
@test sim.t[end] < tspan[2]

# Don't wrap function
test_func = (u, t, integrator) -> DiffEqCallbacks.allDerivPass(integrator, 1e-6, 1e-6,
    nothing)

@test_throws MethodError solve(prob, Tsit5(), callback = TerminateSteadyState(1e-6, 1e-6,
    test_func))

sim = solve(prob, Tsit5(), callback = TerminateSteadyState(1e-6, 1e-6, test_func,
    wrap_test = Val(false)))

@test all(sim(sim.t[end], Val{1}) .< 1e-8)
@test sim.t[end] < tspan[2]
