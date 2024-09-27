using OrdinaryDiffEq, DiffEqCallbacks, Test

function some_dynamics(u, p, t)
    du = zeros(length(u))
    for i in 1:length(u)
        for j in 1:length(u)
            du[i] += p[i, j] * u[j] * u[i]
        end
    end
    return du
end

p = rand(4, 4)
startp = copy(p)

u0 = zeros(length(p[1, :])) .+ 0.1
tspan = (0.0, 1.0)
prob = ODEProblem(some_dynamics, u0, tspan, p)
cb = PresetTimeCallback(0.5, integrator -> integrator.p .= rand(4, 4))
sol = solve(prob, Tsit5(), callback = cb)
@test 0.5 ∈ sol.t
@test p != startp

p = rand(4, 4)
startp = copy(p)

prob = ODEProblem(some_dynamics, u0, tspan, p)
cb = PresetTimeCallback([0.0, 0.3, 0.6], integrator -> integrator.p .= rand(4, 4))
integrator = init(prob, Tsit5(), callback = cb)
@test first_tstop(integrator) == 0.3
solve!(integrator)
sol = integrator.sol
@test 0.3 ∈ sol.t
@test 0.6 ∈ sol.t
@test p != startp

notcalled = true
prob = ODEProblem(some_dynamics, u0, tspan, p)
cb = PresetTimeCallback([1.2], integrator -> notcalled = false)
sol = solve(prob, Tsit5(), callback = cb)
@test notcalled

cb = PresetTimeCallback([1.2], integrator -> begin
        global notcalled
        notcalled = false
    end, filter_tstops = false)
sol = solve(prob, Tsit5(), callback = cb)
@test !notcalled

notcalled = true
prob = ODEProblem(some_dynamics, u0, (1.0, 0.0), p)
cb = PresetTimeCallback([-0.2], integrator -> notcalled = false)
sol = solve(prob, Tsit5(), callback = cb)
@test notcalled

cb = PresetTimeCallback([-0.2], integrator -> begin
        global notcalled
        notcalled = false
    end, filter_tstops = false)
sol = solve(prob, Tsit5(), callback = cb)
@test !notcalled

# Test indexes reset
# https://github.com/SciML/DifferentialEquations.jl/issues/1022

function mod(du, u, p, t)
    du[1] = -p[1] * u[1]
end

p = [1.0]
u0 = [10.0]
tspan = (0.0, 72.0)

times1 = 0.0:24.0:tspan[2]
times2 = 24.0:24.0:tspan[2]
affect!(integrator) = integrator.u[1] += 10.0
cb1 = PresetTimeCallback(times1, affect!)
cb2 = PresetTimeCallback(times2, affect!)

prob1 = ODEProblem(mod, u0, tspan, p, callback = cb1)
prob2 = ODEProblem(mod, u0, tspan, p)

sol1 = solve(prob1, Tsit5())
sol2 = solve(prob2, Tsit5(), callback = cb1)

@test sol1(0.0) == [10.0]
@test sol1(24.0 + eps(24.0)) ≈ [10.0]
@test sol1(48.0 + eps(48.0)) ≈ [10.0]
@test sol2(0.0) == [10.0]
@test sol2(24.0 + eps(24.0)) ≈ [10.0]
@test sol2(48.0 + eps(48.0)) ≈ [10.0]

_some_test_func(integrator) = u_modified!(integrator, false)
@inferred PresetTimeCallback(
    collect(range(0, 10, 100)), _some_test_func, save_positions = (false, false))
