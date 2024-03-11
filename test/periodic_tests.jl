using Test, OrdinaryDiffEq, DiffEqCallbacks

tmin = 0.1
tmax = 5.2

for tmax_problem in [tmax; Inf]
    # Test with both a finite and an infinite tspan for the ODEProblem.
    #
    # Having support for infinite tspans is one of the main reasons for implementing PeriodicCallback
    # using add_tstop!, instead of just passing in a linspace as the tstops solve argument.
    # (the other being that the length of the internal tstops collection would otherwise become
    # linear in the length of the integration time interval.
    #
    # Testing a finite tspan is necessary because a naive implementation could add tstops after
    # tmax and thus integrate for too long (or even indefinitely).

    # Dynamics: two independent single integrators:
    du = [0; 0]
    u0 = [0.0; 0.0]
    dynamics = (u, p, t) -> eltype(u).(du)
    prob = ODEProblem(dynamics, u0, (tmin, tmax_problem))

    # Callbacks periodically increase the input to the integrators:
    Δt1 = 0.5
    increase_du_1 = integrator -> du[1] += 1
    periodic1_initialized = Ref(false)
    initialize1 = (c, u, t, integrator) -> periodic1_initialized[] = true
    periodic1 = PeriodicCallback(increase_du_1, Δt1; initialize = initialize1)

    Δt2 = 1.0
    increase_du_2 = integrator -> du[2] += 1
    periodic2 = PeriodicCallback(increase_du_2, Δt2)

    # Terminate at tmax (regardless of whether the tspan of the ODE problem is infinite).
    terminator = DiscreteCallback((u, t, integrator) -> t == tmax, terminate!)

    # Solve.
    sol = solve(prob, Tsit5(); callback = CallbackSet(terminator, periodic1, periodic2),
        tstops = [tmax])

    # Ensure that initialize1 has been called
    @test periodic1_initialized[]

    # Make sure we haven't integrated past tmax:
    @test sol.t[end] == tmax

    # Make sure that the components of du have been incremented the appropriate number of times.
    Δts = [Δt1, Δt2]
    expected_num_calls = map(Δts) do Δt
        floor(Int, (tmax - tmin) / Δt)
    end
    @test du == expected_num_calls

    # Make sure that the final state matches manual integration of the piecewise linear function
    foreach(Δts, sol.u[end], du) do Δt, u_i, du_i
        @test u_i≈Δt * sum(1:(du_i - 1)) + rem(tmax - tmin, Δt) * du_i atol=1e-5
    end
end

function fff(du, u, p, t)
    du[1] = -u[1]
    du[2] = 0
end

u0 = [2.0, 0.0]
function periodic(integrator)
    integrator.u[2] = integrator.u[1]
end
cb = PeriodicCallback(periodic, 0.1, initial_affect = true, save_positions = (true, true))
tspan = (0.0, 10.0)
p = nothing

prob = ODEProblem(fff, u0, tspan, p)
sol = solve(prob, Tsit5(), callback = cb)
@test sol.u[2] == [2.0, 2.0]
@test sol.u[end][1] != sol.u[end][2] # `final_affect = false` by default

# Test that the callback is applied again when the simulation finished.
cb = PeriodicCallback(periodic, 3.0, initial_affect = true, final_affect = true,
    save_positions = (true, true))
sol = solve(prob, Tsit5(), callback = cb)
@test sol.u[end][1] == sol.u[end][2]

# Test a PeriodicCallback that stops the simulation with terminate!(integrator)
periodic_terminate2 = integrator -> if integrator.t >= tmax
    terminate!(integrator)
end
cb = PeriodicCallback(periodic_terminate2, 0.1, initial_affect = true, final_affect = true,
    save_positions = (true, true))
sol = solve(prob, Tsit5(), callback = cb)
@test sol.retcode == ReturnCode.Terminated
@test sol.t[end] == tmax

# Test that `Δt > tspan[2]` does not extend the simulation beyond `tspan[2]`
# when `initial_affect = false`.
cb = PeriodicCallback(periodic, 11.0, initial_affect = false)
prob = ODEProblem(fff, u0, tspan, p)
sol = solve(prob, Tsit5(), callback = cb)
@test sol.t[end] == tspan[2]


# Fix indexing repeats
# https://github.com/SciML/ModelingToolkit.jl/issues/2528

function lineardecay(du, u, p, t)
    du[1] = -u[1]
end

function bumpaffect!(integ)
    integ.u[1] += 10
end

cb = PeriodicCallback(bumpaffect!, 24.0)
prob = ODEProblem(lineardecay, [0.0], (0.0, 130.0))
sol1 = solve(prob, Tsit5(), callback = cb)

@test sol1(0.0) == [0.0]
@test sol1(24.0 + eps(24.0)) ≈ [10.0]
@test sol1(48.0 + eps(48.0)) ≈ [10.0]
@test sol1(72.0 + eps(72.0)) ≈ [10.0]
@test sol1(96.0 + eps(96.0)) ≈ [10.0]
@test sol1(120.0 + eps(120.0)) ≈ [10.0]
sol2 = solve(prob, Tsit5(), callback = cb)
@test sol2(0.0) == [0.0]
@test sol2(24.0 + eps(24.0)) ≈ [10.0]
@test sol2(48.0 + eps(48.0)) ≈ [10.0]
@test sol2(72.0 + eps(72.0)) ≈ [10.0]
@test sol2(96.0 + eps(96.0)) ≈ [10.0]
@test sol2(120.0 + eps(120.0)) ≈ [10.0]
