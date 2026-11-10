using OrdinaryDiffEqTsit5, DiffEqCallbacks, ForwardDiff, Test

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

cb = PresetTimeCallback(
    [1.2], integrator -> begin
        global notcalled
        notcalled = false
    end, filter_tstops = false
)
sol = solve(prob, Tsit5(), callback = cb)
@test !notcalled

notcalled = true
prob = ODEProblem(some_dynamics, u0, (1.0, 0.0), p)
cb = PresetTimeCallback([-0.2], integrator -> notcalled = false)
sol = solve(prob, Tsit5(), callback = cb)
@test notcalled

cb = PresetTimeCallback(
    [-0.2], integrator -> begin
        global notcalled
        notcalled = false
    end, filter_tstops = false
)
sol = solve(prob, Tsit5(), callback = cb)
@test !notcalled

# Test indexes reset
# https://github.com/SciML/DifferentialEquations.jl/issues/1022

function mod(du, u, p, t)
    return du[1] = -p[1] * u[1]
end

p = [1.0]
u0 = [10.0]
tspan = (0.0, 72.0)

times1 = 0.0:24.0:tspan[2]
times2 = 24.0:24.0:tspan[2]
preset_affect!(integrator) = integrator.u[1] += 10.0
cb1 = PresetTimeCallback(times1, preset_affect!)
cb2 = PresetTimeCallback(times2, preset_affect!)

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

_some_test_func(integrator) = derivative_discontinuity!(integrator, false)
@inferred PresetTimeCallback(
    collect(range(0, 10, 100)), _some_test_func, save_positions = (false, false)
)

# Callable tstops depending on the model parameters
# https://github.com/SciML/DiffEqCallbacks.jl/issues/178
decay_preset!(du, u, p, t) = (du[1] = -u[1])

hits = Float64[]
cb = PresetTimeCallback(integrator -> integrator.p, integrator -> push!(hits, integrator.t))
@inferred PresetTimeCallback(integrator -> integrator.p, integrator -> nothing)
prob = ODEProblem(decay_preset!, [1.0], (0.0, 1.0), [0.6, 0.3])
sol = solve(prob, Tsit5(), callback = cb)
@test hits == [0.3, 0.6]
@test 0.3 ∈ sol.t && 0.6 ∈ sol.t
@test prob.p == [0.6, 0.3]

# The same callback picks up new stops after a remake with updated parameters.
empty!(hits)
sol = solve(remake(prob; p = [0.75]), Tsit5(), callback = cb)
@test hits == [0.75]
@test 0.75 ∈ sol.t

# A callable may return a single time, and stops outside the interval are filtered.
empty!(hits)
cb_scalar = PresetTimeCallback(
    integrator -> integrator.p[1], integrator -> push!(hits, integrator.t)
)
solve(remake(prob; p = [0.4]), Tsit5(), callback = cb_scalar)
@test hits == [0.4]
empty!(hits)
solve(remake(prob; p = [1.2]), Tsit5(), callback = cb_scalar)
@test isempty(hits)

@test_throws ArgumentError PresetTimeCallback("0.5", integrator -> nothing)
cb_bad = PresetTimeCallback(integrator -> "0.5", integrator -> nothing)
@test_throws ArgumentError solve(prob, Tsit5(), callback = cb_bad)

# ForwardDiff through a parameter-dependent stop: u' = -u with u += 1 at
# t = p[1] gives u(1) = exp(-1) + exp(-(1 - p[1])).
function preset_fd_solve(pval)
    cb_fd = PresetTimeCallback(
        integrator -> integrator.p, integrator -> (integrator.u[1] += 1)
    )
    prob_fd = ODEProblem(decay_preset!, [1 + 0 * pval], (0 * pval, 1 + 0 * pval), [pval])
    sol_fd = solve(
        prob_fd, Tsit5(), callback = cb_fd, save_everystep = false,
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    return sol_fd.u[end][1]
end
@test preset_fd_solve(0.4) ≈ exp(-1) + exp(-0.6) atol = 1.0e-8
@test ForwardDiff.derivative(preset_fd_solve, 0.4) ≈ exp(-0.6) atol = 1.0e-6
