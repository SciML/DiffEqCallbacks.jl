using OrdinaryDiffEqTsit5, DiffEqCallbacks, Test
using SciMLBase: SciMLBase

# Ticks recorded as (t, prev, next, sign of x at the tick)
function crossing_ticks(
        x, tspan; f = (u, p, t) -> [0.0], u0 = [0.0], p = nothing,
        before = nothing, after = nothing, solve_kwargs = (; dtmax = 0.1), kwargs...
    )
    ticks = Tuple{Float64, Int, Int, Int}[]
    affect! = (integrator, prev, next) -> push!(
        ticks,
        (integrator.t, prev, next, Int(sign(x(integrator.u, integrator.t, integrator))))
    )
    cb = CrossingCallback(x, affect!; kwargs...)
    cb = before === nothing ? cb : CallbackSet(before, cb)
    cb = after === nothing ? cb : CallbackSet(cb, after)
    sol = solve(ODEProblem(f, u0, tspan, p), Tsit5(); callback = cb, solve_kwargs...)
    @test SciMLBase.successful_retcode(sol)
    return ticks
end

# One float beyond `t` in the direction of integration
beyond(t, dir) = dir > 0 ? nextfloat(t) : prevfloat(t)

# Landing at 1, return at 2, crossing (a jump, so never exactly zero) at 4, landing at 6,
# continuation at 7
function piecewise(u, t, integrator)
    return if t < 1
        t - 1
    elseif t <= 2
        0.0
    elseif t < 4
        2 - t
    elseif t < 6
        6 - t
    elseif t <= 7
        0.0
    else
        7 - t
    end
end

@testset "right = true" begin
    ticks = crossing_ticks(piecewise, (0.0, 8.0))
    @test length(ticks) == 4
    @test ticks[1] == (1.0, -1, 0, 0)
    @test ticks[2] == (nextfloat(2.0), 0, -1, -1)
    @test ticks[3] == (4.0, -1, 1, 1)
    @test ticks[4] == (6.0, 1, 0, 0)
end

@testset "right = false" begin
    ticks = crossing_ticks(piecewise, (0.0, 8.0); right = false)
    @test length(ticks) == 3
    @test ticks[1] == (prevfloat(1.0), -1, 0, -1)
    @test ticks[2] == (prevfloat(4.0), -1, 1, -1)
    @test ticks[3] == (prevfloat(6.0), 1, 0, 1)
end

@testset "crossing through an exact zero" begin
    for tspan in ((0.0, 4.0), (4.0, 0.0))
        dir = Int(sign(tspan[2] - tspan[1]))
        x = (u, t, integrator) -> t - 2.0
        @test crossing_ticks(x, tspan) == [(2.0, -dir, dir, 0)]
        before = dir > 0 ? prevfloat(2.0) : nextfloat(2.0)
        @test crossing_ticks(x, tspan; right = false) == [(before, -dir, dir, -dir)]
    end
end

@testset "start from zero" begin
    x = (u, t, integrator) -> t <= 1 ? 0.0 : 1 - t
    @test crossing_ticks(x, (0.0, 2.0)) == [(nextfloat(1.0), 0, -1, -1)]
    @test crossing_ticks(x, (0.0, 2.0); right = false) == [(1.0, 0, -1, 0)]
end

@testset "initial" begin
    x = (u, t, integrator) -> cos(t)
    ticks = crossing_ticks(x, (0.0, 2.0); initial = true)
    @test ticks[1] == (0.0, 0, 1, 1)
    @test length(ticks) == 2
    @test isempty(crossing_ticks(x, (0.0, 1.0)))
end

@testset "backward" begin
    x = (u, t, integrator) -> cos(t)
    ticks = crossing_ticks(x, (5.0, 0.0))
    @test [tick[1] for tick in ticks] ≈ [3π / 2, π / 2]
    @test [tick[2:4] for tick in ticks] == [(1, -1, -1), (-1, 1, 1)]
    ticks = crossing_ticks(x, (5.0, 0.0); right = false)
    @test [tick[1] for tick in ticks] ≈ [3π / 2, π / 2]
    @test [tick[2:4] for tick in ticks] == [(1, -1, 1), (-1, 1, -1)]
end

@testset "state crossing" begin
    # u = sin(t): starts at zero, crosses at π and 2π
    x = (u, t, integrator) -> u[1]
    for right in (true, false)
        ticks = crossing_ticks(x, (0.0, 7.0); f = (u, p, t) -> [cos(t)], right)
        @test [tick[2:3] for tick in ticks] == [(0, 1), (1, -1), (-1, 1)]
        @test ticks[1][1] <= 1.0e-12
        @test ticks[2][1] ≈ π atol = 1.0e-4
        @test ticks[3][1] ≈ 2π atol = 1.0e-4
        @test all(tick -> tick[4] == (right ? tick[3] : tick[2]), ticks)
    end
end

@testset "stick and slip" begin
    # Another callback stops the motion at 0.5 and restarts it backward at 1.5
    x = (u, t, integrator) -> u[1]
    f = (u, p, t) -> [p[1]]
    stick = PresetTimeCallback(0.5, integrator -> (integrator.u[1] = 0.0; integrator.p[1] = 0.0))
    slip = PresetTimeCallback(1.5, integrator -> (integrator.p[1] = -1.0))
    ticks = crossing_ticks(
        x, (0.0, 2.0); f, u0 = [-1.0], p = [1.0], before = CallbackSet(stick, slip)
    )
    @test ticks[1] == (0.5, -1, 0, 0)
    @test ticks[2][1] ≈ 1.5 && ticks[2][1] > 1.5
    @test ticks[2][2:4] == (0, -1, -1)
    @test length(ticks) == 2

    ticks = crossing_ticks(
        x, (0.0, 2.0); f, u0 = [-1.0], p = [1.0], before = CallbackSet(stick, slip),
        right = false
    )
    @test ticks == [(0.5, -1, 0, 0)]
end

@testset "several roots in one step" begin
    # A tick must not hide the roots that follow it within the same step
    x = (u, t, integrator) -> (t - 1) * (t - 2) * (t - 3)
    for tspan in ((0.0, 4.0), (4.0, 0.0))
        dir = Int(sign(tspan[2] - tspan[1]))
        roots = dir > 0 ? (1.0, 2.0, 3.0) : (3.0, 2.0, 1.0)
        signs = (-dir, dir, -dir, dir)
        right_ticks = [(roots[i], signs[i], signs[i + 1], 0) for i in 1:3]
        left_ticks = [(beyond(roots[i], -dir), signs[i], signs[i + 1], signs[i]) for i in 1:3]
        for (solve_kwargs, kwargs) in (
                ((; adaptive = false, dt = 2.5), (; interp_points = 100)),
                ((;), (;)),
            )
            @test crossing_ticks(x, tspan; solve_kwargs, kwargs...) == right_ticks
            @test crossing_ticks(x, tspan; solve_kwargs, right = false, kwargs...) ==
                left_ticks
        end
    end
end

@testset "solve ending at a root" begin
    # Nothing is known beyond the end, so reaching zero there is a landing
    x = (u, t, integrator) -> t - 1
    for tspan in ((0.0, 1.0), (2.0, 1.0))
        dir = Int(sign(tspan[2] - tspan[1]))
        @test crossing_ticks(x, tspan) == [(1.0, -dir, 0, 0)]
        @test crossing_ticks(x, tspan; right = false) == [(beyond(1.0, -dir), -dir, 0, -dir)]
    end
end

@testset "end time after reinit!" begin
    ticks = Tuple{Float64, Int, Int}[]
    cb = CrossingCallback(
        (u, t, integrator) -> t - 2, (integrator, prev, next) -> push!(ticks, (integrator.t, prev, next))
    )
    integrator = init(ODEProblem((u, p, t) -> [0.0], [0.0], (0.0, 1.0)), Tsit5(); callback = cb, dtmax = 0.1)
    reinit!(integrator, [0.0]; tf = 4.0)
    solve!(integrator)
    @test ticks == [(2.0, -1, 1)]
end

@testset "recrossing right after a crossing" begin
    # Sign-valued x: the next step must not skip its start, about dt/100 = 1e-3, as the root
    # just handled. The interpolation points are dense enough to see the gap.
    x = (u, t, integrator) -> sign((t - 1) * (t - 1.0005))
    @test crossing_ticks(x, (0.0, 2.0); interp_points = 1000) ==
        [(1.0, 1, -1, 0), (1.0005, -1, 1, 0)]
end

@testset "affect! that reverses the motion" begin
    # A bouncing ball: with right = false the tick is before the floor, and the reflected
    # ball never goes below it, so there is no tick back up
    for right in (false, true)
        ticks = Tuple{Float64, Int, Int}[]
        affect! = function (integrator, prev, next)
            push!(ticks, (integrator.t, prev, next))
            prev > 0 && (integrator.u[2] = -integrator.u[2])
            return nothing
        end
        cb = CrossingCallback((u, t, integrator) -> u[1], affect!; right)
        prob = ODEProblem((u, p, t) -> [u[2], -9.81], [1.0, 0.0], (0.0, 2.0))
        sol = solve(prob, Tsit5(); callback = cb, abstol = 1.0e-10, reltol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        impact = sqrt(2 / 9.81)
        if right
            # Beyond the floor, the reflected ball crosses back up
            @test [tick[2:3] for tick in ticks] == [(1, -1), (-1, 1), (1, -1), (-1, 1)]
        else
            @test [tick[2:3] for tick in ticks] == [(1, -1), (1, -1)]
            @test minimum(sol[1, :]) >= 0
        end
        @test ticks[1][1] ≈ impact atol = 1.0e-6
        @test ticks[end - right][1] ≈ 3impact atol = 1.0e-6
    end
end

@testset "change made after CrossingSync" begin
    # Another callback, after this one in the CallbackSet, stops the motion at 0.5 and
    # restarts it backward at 1.5: the landing is found just after 0.5, the return at 1.5
    x = (u, t, integrator) -> u[1]
    f = (u, p, t) -> [p[1]]
    stick = PresetTimeCallback(0.5, integrator -> (integrator.u[1] = 0.0; integrator.p[1] = 0.0))
    slip = PresetTimeCallback(1.5, integrator -> (integrator.p[1] = -1.0))
    ticks = crossing_ticks(
        x, (0.0, 2.0); f, u0 = [-1.0], p = [1.0], after = CallbackSet(stick, slip)
    )
    @test length(ticks) == 2
    @test 0.5 < ticks[1][1] <= 0.5 + 1.0e-12 && ticks[1][2:4] == (-1, 0, 0)
    @test 1.5 < ticks[2][1] <= 1.5 + 1.0e-12 && ticks[2][2:4] == (0, -1, -1)
end

@testset "NaN and reserved keywords" begin
    # A NaN x is no sign change
    @test isempty(crossing_ticks((u, t, integrator) -> t < 1 ? 1.0 : NaN, (0.0, 2.0)))
    for k in (:initialize, :rootfind, :affect_neg!, :idxs)
        @test_throws ArgumentError CrossingCallback(
            (u, t, integrator) -> t, (integrator, prev, next) -> nothing; (k => nothing,)...
        )
    end
end

# The sign held after a tick by a consumer that keeps the direction of a crossing as the
# sign of a function that lands on zero
held_sign(prev, next) = iszero(next) ? -prev : next

@testset "exact zeros" begin
    tight = (; abstol = 1.0e-10, reltol = 1.0e-10, dtmax = 0.05)

    @testset "leaves zero at the start" begin
        # x = t^2/2 - t^3: zero at 0, positive right after, crosses down at 1/2
        ticks = crossing_ticks(
            (u, t, integrator) -> u[1], (0.0, 1.0); f = (u, p, t) -> [u[2], 1 - 6t],
            u0 = [0.0, 0.0], solve_kwargs = tight
        )
        @test [tick[2:3] for tick in ticks] == [(0, 1), (1, -1)]
        @test ticks[1][1] <= 1.0e-12
        @test ticks[2][1] ≈ 0.5 atol = 1.0e-6
    end

    @testset "direction known before a later tick" begin
        # w' = -1 from w(0) = 0
        ticks = crossing_ticks((u, t, integrator) -> u[1], (0.0, 1.0); f = (u, p, t) -> [-1.0])
        @test length(ticks) == 1
        @test ticks[1][1] <= 1.0e-12 && ticks[1][2:4] == (0, -1, -1)
    end

    @testset "derivative zero at initialization" begin
        # w' = -p[1], with p[1] switched from 0 to 1 at the initial tick
        on = DiscreteCallback(
            Returns(false), Returns(nothing); save_positions = (false, false),
            initialize = (c, u, t, integrator) -> (integrator.p[1] = 1.0; SciMLBase.derivative_discontinuity!(integrator, true))
        )
        ticks = crossing_ticks(
            (u, t, integrator) -> u[1], (0.0, 1.0); f = (u, p, t) -> [-p[1]], p = [0.0],
            before = on
        )
        @test length(ticks) == 1
        @test ticks[1][1] <= 1.0e-12 && ticks[1][2:4] == (0, -1, -1)
    end

    @testset "step onto a zero plateau and off it" begin
        fx = (u, t, integrator) -> t < 1 ? 0.0 : t < 2 ? t - 1 : t < 3 ? 0.0 : t - 3
        ticks = crossing_ticks(fx, (0.0, 4.0); initial = true)
        @test ticks == [
            (0.0, 0, 0, 0), (nextfloat(1.0), 0, 1, 1), (2.0, 1, 0, 0), (nextfloat(3.0), 0, 1, 1),
        ]
        @test [held_sign(tick[2], tick[3]) > 0 for tick in ticks] == [false, true, false, true]
    end

    @testset "smooth approach to a plateau" begin
        # The entry is located at exactly 1, not somewhere in the step
        gx = (u, t, integrator) -> t < 1 ? 1 - t : t < 2 ? 0.0 : t - 2
        ticks = crossing_ticks(gx, (0.0, 4.0); initial = true)
        @test ticks == [(0.0, 0, 1, 1), (1.0, 1, 0, 0), (nextfloat(2.0), 0, 1, 1)]
        @test [held_sign(tick[2], tick[3]) > 0 for tick in ticks] == [true, false, true]
        # With `right = false` the landing is still recognized
        ticks = crossing_ticks(gx, (0.0, 4.0); initial = true, right = false)
        @test ticks[1:2] == [(0.0, 0, 1, 1), (prevfloat(1.0), 1, 0, 1)]
    end
end
