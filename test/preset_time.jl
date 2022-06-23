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
cb = PresetTimeCallback([0.3, 0.6], integrator -> integrator.p .= rand(4, 4))
sol = solve(prob, Tsit5(), callback = cb)
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
