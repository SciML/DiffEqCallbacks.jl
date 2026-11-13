using OrdinaryDiffEqTsit5, DiffEqCallbacks, Test

# Ticks recorded as (t, prev, next, sign of x at the tick)
function crossing_ticks(x, tspan; f = (u, p, t) -> [0.0], u0 = [0.0], p = nothing,
        before = nothing, kwargs...)
    ticks = Tuple{Float64, Int, Int, Int}[]
    affect! = (integrator, prev, next) -> push!(
        ticks,
        (integrator.t, prev, next, Int(sign(x(integrator.u, integrator.t, integrator))))
    )
    cb = CrossingCallback(x, affect!; kwargs...)
    cb = before === nothing ? cb : CallbackSet(before, cb)
    solve(ODEProblem(f, u0, tspan, p), Tsit5(); callback = cb, dtmax = 0.1)
    return ticks
end

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
