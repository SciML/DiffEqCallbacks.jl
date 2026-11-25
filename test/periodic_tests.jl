using Test, OrdinaryDiffEqTsit5, DiffEqCallbacks
using OrdinaryDiffEqLowOrderRK: Euler
using SciMLBase: EnsembleProblem, EnsembleThreads, remake, ReturnCode

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
    local du = [0; 0]
    local u0 = [0.0; 0.0]
    dynamics = (u, p, t) -> eltype(u).(du)
    local prob = ODEProblem(dynamics, u0, (tmin, tmax_problem))

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
    local sol = solve(
        prob, Tsit5(); callback = CallbackSet(terminator, periodic1, periodic2),
        tstops = [tmax]
    )

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
        @test u_i ≈ Δt * sum(1:(du_i - 1)) + rem(tmax - tmin, Δt) * du_i atol = 1.0e-5
    end
end

function fff(du, u, p, t)
    du[1] = -u[1]
    return du[2] = 0
end

u0 = [2.0, 0.0]
function periodic(integrator)
    return integrator.u[2] = integrator.u[1]
end
cb = PeriodicCallback(periodic, 0.1, initial_affect = true, save_positions = (true, true))
tspan = (0.0, 10.0)
p = nothing

prob = ODEProblem(fff, u0, tspan, p)
sol = solve(prob, Tsit5(), callback = cb)
@test sol.u[2] == [2.0, 2.0]
@test sol.u[end][1] != sol.u[end][2] # `final_affect = false` by default

# Test that the callback is applied again when the simulation finished.
cb = PeriodicCallback(
    periodic, 3.0, initial_affect = true, final_affect = true,
    save_positions = (true, true)
)
sol = solve(prob, Tsit5(), callback = cb)
@test sol.u[end][1] == sol.u[end][2]

# Test a PeriodicCallback that stops the simulation with terminate!(integrator)
periodic_terminate2 = integrator -> if integrator.t >= tmax
    terminate!(integrator)
end
cb = PeriodicCallback(
    periodic_terminate2, 0.1, initial_affect = true, final_affect = true,
    save_positions = (true, true)
)
sol = solve(prob, Tsit5(), callback = cb)
@test sol.retcode == ReturnCode.Terminated
@test sol.t[end] == tmax

# Test that `Δt > tspan[2]` does not extend the simulation beyond `tspan[2]`
# when `initial_affect = false`.
cb = PeriodicCallback(periodic, 11.0, initial_affect = false)
prob = ODEProblem(fff, u0, tspan, p)
sol = solve(prob, Tsit5(), callback = cb)
@test sol.t[end] == tspan[2]

# Rational Δt must fire periodically (DifferentialEquations.jl #878).
# Exact `t == t0 + index*Δt` fails when t is Float64 and Δt is Rational.
@testset "Rational PeriodicCallback period" begin
    firings = Float64[]
    cb_rat = PeriodicCallback(
        integrator -> push!(firings, integrator.t), 1 // 10;
        initial_affect = true
    )
    prob_rat = ODEProblem((du, u, p, t) -> (du[1] = -u[1]), [1.0], (0.0, 1.0))
    sol_rat = solve(prob_rat, Tsit5(); callback = cb_rat)
    @test sol_rat.retcode == ReturnCode.Success
    # t = 0, 0.1, ..., 1.0 → 11 firings (initial + 10 periods; final tstop may or
    # may not double-fire depending on tstop handling — require at least the
    # interior grid).
    @test length(firings) >= 10
    @test firings[1] == 0.0
    @test all(isapprox(firings[i], (i - 1) / 10; atol = 1.0e-12) for i in eachindex(firings) if firings[i] <= 1.0)
end

# Coincident ticks of periodic callbacks with rational periods and phases are
# equal as floating-point numbers, while the same ticks computed with floating-point
# periods differ by one ulp.
@testset "Rational PeriodicCallback phase and coincident ticks" begin
    prob_rat = ODEProblem((du, u, p, t) -> (du[1] = -u[1]), [1.0], (0.0, 2.0))
    function tick_times(Δt, phase, tspan = prob_rat.tspan)
        ticks = Float64[]
        cb = PeriodicCallback(i -> push!(ticks, i.t), Δt; phase)
        solve(remake(prob_rat; tspan), Tsit5(); callback = cb)
        return ticks
    end
    @test 270 * 0.005 != 1 / 60 + 40 * (1 / 30)
    # Both callbacks in one integrator: the coincident tick is one tstop, and
    # no step between the two ticks is taken.
    ticks1, ticks2 = Float64[], Float64[]
    cb1 = PeriodicCallback(i -> push!(ticks1, i.t), 1 // 200)
    cb2 = PeriodicCallback(i -> push!(ticks2, i.t), 1 // 30; phase = 1 // 60)
    sol = solve(prob_rat, Tsit5(); callback = CallbackSet(cb1, cb2))
    @test 1.35 in ticks1
    @test 1.35 in ticks2
    @test isempty(intersect(ticks2, prevfloat.(ticks1)) ∪ intersect(ticks2, nextfloat.(ticks1)))
    @test all(Δ -> iszero(Δ) || Δ > 1.0e-6, diff(sol.t))
    @test ticks2[1:3] == Float64.([1 // 60, 1 // 60 + 1 // 30, 1 // 60 + 2 // 30])
    # Each tick is the nearest Float64 to start time + phase + k*Δt, ties to even.
    exact(x) = Rational{BigInt}(x)
    function nearest(x)
        c = (prevfloat(Float64(x)), Float64(x), nextfloat(Float64(x)))
        dist = map(y -> abs(exact(y) - x), c)
        return argmin(i -> (dist[i], isodd(reinterpret(UInt64, c[i]))), 1:3) |> i -> c[i]
    end
    exact_tick(t0, phase, k, Δt) = nearest(exact(t0) + exact(phase) + k * exact(Δt))
    t_start = nextfloat(1.0)
    ticks3 = tick_times(1 // 10, 1 // 2, (t_start, 2.0))
    @test ticks3[1] == t_start + 0.5
    # A start time without a small rational representation does not overflow. The
    # literal is an Int64 also on 32-bit systems, where 10^12 overflows Int32.
    ticks4 = tick_times(1 // 1, 1 // 1_000_000_000_000, (sqrt(2.0), 4.0))
    @test length(ticks4) == 3
    @test ticks4[1] == exact_tick(sqrt(2.0), 1 // 1_000_000_000_000, 0, 1 // 1)
    # A Float64 phase with a rational period takes the exact path: a zero phase
    # (as ModelingToolkit passes it for a clock without phase) and a phase equal to
    # a rational number give the same ticks as the rational phase.
    ticks5 = tick_times(1 // 10, 0.0, (0.3, 1.0))
    @test ticks5[1] === exact_tick(0.3, 0, 1, 1 // 10) === 0.39999999999999997
    ticks6 = tick_times(1 // 10, 0.5, (0.1, 1.0))
    @test ticks6 == tick_times(1 // 10, 1 // 2, (0.1, 1.0))
    # A Float64 phase is converted exactly, without overflow of Rational{Int}.
    ticks7 = tick_times(1 // 1_000_000_000_000, 0.1, (sqrt(2.0), sqrt(2.0) + 0.1 + 1.0e-11))
    @test ticks7[1] == exact_tick(sqrt(2.0), 0.1, 0, 1 // 1_000_000_000_000)
    # The increments survive cancellation of the start time and the phase.
    ticks8 = Float64[]
    cb8 = PeriodicCallback(i -> push!(ticks8, i.t), 1 // Int64(10)^16; phase = 1)
    solve(
        remake(prob_rat; tspan = (-1.0, 3.5e-16)), Tsit5(); callback = cb8,
        adaptive = false, dt = 1.0
    )
    @test ticks8 == Float64.((0:3) .// Int64(10)^16)
    # Equal periods stored with different integer types tick at the same time;
    # `Float64(::Rational{Int})` would round the denominator first and give
    # 0.5000000000000001.
    p = (Int64(2)^52 + 1) // (Int64(2)^53 + 1)
    ticks9, ticks10 = Float64[], Float64[]
    sol = solve(
        remake(prob_rat; tspan = (0.0, 0.75)), Tsit5();
        callback = CallbackSet(
            PeriodicCallback(i -> push!(ticks9, i.t), p),
            PeriodicCallback(i -> push!(ticks10, i.t), big(p))
        )
    )
    @test ticks9 == ticks10 == [0.5]
    @test all(Δ -> iszero(Δ) || Δ > 1.0e-6, diff(sol.t))
    # Narrow integer types do not overflow.
    @test tick_times(Int32(1) // Int32(65537), Int32(1) // Int32(65536), (0.0, 5.0e-5)) ==
        [exact_tick(0.0, 1 // 65536, k, 1 // 65537) for k in 0:2]
    @test tick_times(1 // 3, (Int64(2)^61 + 1) // Int64(2)^62, (0.0, 1.0)) ==
        [exact_tick(0.0, (big(2)^61 + 1) // big(2)^62, k, 1 // 3) for k in 0:1]
    # A floating-point phase with an integer period is added to the start time first.
    @test tick_times(1, 0.5, (0.5, 2.5)) == [1.0, 2.0]
    # The conversion of a tick to Float64 rounds once and does not depend on the
    # BigFloat precision.
    @test setprecision(() -> tick_times(1 // 10, 0, (0.0, nextfloat(0.1))), BigFloat, 24) ==
        [0.1]
    # The period is the midpoint of the two smallest Float64 numbers above 0.5. The
    # tick, one period after the slightly negative start time, is the lower neighbor;
    # rounding the midpoint to even gives the end time instead.
    p = (Int64(2)^53 + 3) // Int64(2)^54
    t_start = -0x1p-300
    ticks11 = tick_times(p, 0, (t_start, nextfloat(0.5, 2)))
    @test ticks11 == [exact_tick(t_start, 0, 1, p)] == [nextfloat(0.5)]
end

@testset "Rational PeriodicCallback with an infinite scheduled time" begin
    for (Δt, phase, tspan) in (
            (1 // 4, Inf, (0.0, 1.0)),
            (1 // 0, 0, (0.0, 1.0)),
            (-1 // 0, 0, (1.0, 0.0)),
        )
        ticks = Float64[]
        prob = ODEProblem((u, p, t) -> zero(u), 1.0, tspan)
        cb = PeriodicCallback(i -> push!(ticks, i.t), Δt; phase)
        sol = solve(prob, Tsit5(); callback = cb)
        @test sol.retcode == ReturnCode.Success
        @test sol.t[end] == tspan[2]
        @test isempty(ticks)
    end
end

# The first tick with a nonzero phase is computed without the preceding index.
@testset "PeriodicCallback with an unsigned rational time" begin
    ticks = Rational{UInt}[]
    prob = ODEProblem((u, p, t) -> zero(u), 1.0, (UInt(0) // UInt(1), UInt(2) // UInt(1)))
    cb = PeriodicCallback(i -> push!(ticks, i.t), 2; phase = 1)
    sol = solve(prob, Euler(); dt = UInt(1) // UInt(1), callback = cb)
    @test sol.retcode == ReturnCode.Success
    @test ticks == [1]
end

@testset "Rational PeriodicCallback with dual-number time" begin
    using ForwardDiff: Dual, partials, value
    ticks = []
    prob = ODEProblem((u, p, t) -> zero(u), 1.0, (Dual(0.0, 1.0), Dual(0.75, 1.0)))
    cb = PeriodicCallback(i -> push!(ticks, i.t), 1 // 4)
    sol = solve(prob, Tsit5(); callback = cb)
    @test sol.retcode == ReturnCode.Success
    @test value.(ticks) == [0.25, 0.5]
    @test all(t -> partials(t)[1] == 1, ticks)
end

@testset "PeriodicCallback with a BigFloat time" begin
    ticks = BigFloat[]
    t0 = big(2.0)^53 + 1
    prob = ODEProblem((u, p, t) -> zero(u), big(1.0), (t0, t0 + 3))
    cb = PeriodicCallback(i -> push!(ticks, i.t), 1.0; phase = 0.5)
    sol = solve(prob, Euler(); dt = big(0.5), callback = cb)
    @test sol.retcode == ReturnCode.Success
    @test ticks == t0 .+ [0.5, 1.5, 2.5]
    # A rational tick is rounded once to the nearest BigFloat; rounding the
    # numerator first places it at the end time.
    setprecision(BigFloat, 24) do
        ticks = BigFloat[]
        expected = ldexp(BigFloat(13981014), -23)
        tstart = nextfloat(BigFloat(1))
        prob = ODEProblem((u, p, t) -> zero(u), big(1.0), (tstart, nextfloat(expected)))
        cb = PeriodicCallback(i -> push!(ticks, i.t), 1 // 1; phase = 2 // 3)
        sol = solve(prob, Euler(); dt = BigFloat(0.25), callback = cb)
        @test sol.retcode == ReturnCode.Success
        @test ticks == [expected]
    end
end

# A phase that is neither an integer, a rational nor a floating-point number.
@testset "Rational PeriodicCallback with an irrational phase" begin
    ticks = Rational{Int64}[]
    tspan = (Rational{Int64}(0), Rational{Int64}(4))
    prob = ODEProblem((u, p, t) -> zero(u), 1.0, tspan)
    cb = PeriodicCallback(i -> push!(ticks, i.t), big(1) // 1; phase = π)
    sol = solve(prob, Euler(); dt = Rational{Int64}(1, 2), callback = cb)
    @test sol.retcode == ReturnCode.Success
    @test ticks == [Rational{Int64}(Float64(π))]
end

# Fix indexing repeats
# https://github.com/SciML/ModelingToolkit.jl/issues/2528

function lineardecay(du, u, p, t)
    return du[1] = -u[1]
end

function bumpaffect!(integ)
    return integ.u[1] += 10
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

# Test phase offset
approxin(needle, haystack) = any(isapprox(needle), haystack)

function integ(du, u, p, t)
    du[1] = 1
    return du[2] = 0
end

function nullaffect!(integ)
    return integ.u[2] += 1
end

cb = PeriodicCallback(nullaffect!, 1.0; phase = 0.1)
prob = ODEProblem(integ, [0.0, 0], (0.0, 10.0))
sol = solve(prob, Tsit5(), callback = cb)
@test sol.u[end][2] == 10 # Test expected number of calls to affect
for t in 0.1:1:10
    @test approxin(t, sol.t) # Test that the integrator stopped at all expected points
end
sol.t[end] ≈ 10 # test that we did not step over end

# With negative phase
@test_throws ArgumentError PeriodicCallback(nullaffect!, 1.0; phase = -0.1)

# Phase offset larger than period
cb = PeriodicCallback(nullaffect!, 1.0; phase = 1.1)
prob = ODEProblem(integ, [0.0, 0], (0.0, 10.0))
sol = solve(prob, Tsit5(), callback = cb)
@test sol.u[end][2] == 9 # Test expected number of calls to affect
for t in 1.1:1:10
    @test approxin(t, sol.t) # Test that the integrator stopped at all expected points
end
sol.t[end] ≈ 10 # test that we did not step over end

# Re-initialization resets the cache: re-using the same callback on a new
# problem must start from a fresh index/t0, since `initialize_periodic`
# allocates new Refs. Regression test for #99.
@testset "PeriodicCallback re-initializes cache" begin
    cb_reinit = PeriodicCallback(integ -> nothing, 0.5)
    prob_reinit = ODEProblem((u, p, t) -> u, [1.0], (0.0, 2.0))
    sol_a = solve(prob_reinit, Tsit5(), callback = cb_reinit)
    sol_b = solve(prob_reinit, Tsit5(), callback = cb_reinit)
    @test sol_a.t == sol_b.t
end

# Thread safety: because the cache lives in `task_local_storage`, a single
# callback object shared across `EnsembleThreads` trajectories must not leak
# state between Tasks. Regression test for #99.
@testset "PeriodicCallback is EnsembleThreads-safe" begin
    thread_dynamics = (du, u, p, t) -> (du[1] = 1; nothing)
    prob_thread = ODEProblem(thread_dynamics, [0.0], (0.0, 10.0))
    cb_thread = PeriodicCallback(integ -> nothing, 0.1)
    ensemble = EnsembleProblem(
        prob_thread,
        prob_func = (prob, ctx_or_i, _...) -> begin
            sim_id = ctx_or_i isa Integer ? ctx_or_i : ctx_or_i.sim_id
            remake(prob; u0 = [Float64(sim_id)])
        end
    )
    for _ in 1:10
        sol_thread = solve(
            ensemble, Tsit5(), EnsembleThreads();
            trajectories = 16, callback = cb_thread
        )
        @test length(sol_thread.u) == 16
        @test all(s -> s.retcode == ReturnCode.Success, sol_thread.u)
        @test all(s -> s.t[end] == 10.0, sol_thread.u)
    end
end
