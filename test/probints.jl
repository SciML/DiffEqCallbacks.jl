using DiffEqCallbacks, DiffEqBase, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5
using ForwardDiff, LinearAlgebra, OrdinaryDiffEqRosenbrock, Random, RecursiveArrayTools, SciMLBase,
    StaticArrays, Test

function g(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    return du[3] = x * y - β * z
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 10.0)
p = [10.0, 28.0, 8 / 3]
prob = ODEProblem(g, u0, tspan, p)

cb = ProbIntsUncertainty(1.0e4, 5)
solve(prob, Tsit5())
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 10, callback = cb, adaptive = false,
    dt = 1 / 10
)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

function fitz(du, u, p, t)
    V, R = u
    du[1] = 3.0 * (V - V^3 / 3 + R)
    return du[2] = -(1 / 3.0) * (V - 0.2 - 0.2 * R)
end
u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(fitz, u0, tspan)

cb = ProbIntsUncertainty(0.1, 1)
sol = solve(prob, Euler(), dt = 1 / 10)
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Euler(), trajectories = 100, callback = cb, adaptive = false,
    dt = 1 / 10
)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

cb = AdaptiveProbIntsUncertainty(5)
sol = solve(prob, Tsit5())
monte_prob = EnsembleProblem(prob)
sim = solve(
    monte_prob, Tsit5(), trajectories = 100, callback = cb, abstol = 1.0e-3,
    reltol = 1.0e-1
)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

function _probints_affect_bytes(alg, cb)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        g, [1.0, 0.0, 0.0], (0.0, 1.0), [10.0, 28.0, 8 / 3]
    )
    integ = init(prob, alg; callback = cb, save_everystep = false)
    step!(integ)
    return _probints_callback_bytes(cb.affect!, integ)
end

function _probints_callback_bytes(affect!, integ)
    affect!(integ)
    return @allocated affect!(integ)
end

function _reference_probints(σ, order)
    affect! = function (integrator)
        return integrator.u .= integrator.u .+ σ * sqrt(integrator.dt^(2 * order)) * randn(size(integrator.u))
    end
    return DiscreteCallback((u, t, integrator) -> true, affect!; save_positions = (true, false))
end

function _reference_adaptive_probints(order)
    affect! = function (integrator)
        eest = if hasproperty(integrator, :EEst)
            integrator.EEst
        else
            integrator.controller_cache.EEst
        end
        return integrator.u .= integrator.u .+ eest * sqrt(integrator.dt^(2 * order)) * randn(size(integrator.u))
    end
    return DiscreteCallback((u, t, integrator) -> true, affect!; save_positions = (true, false))
end

@testset "Seeded ProbInts solutions" begin
    for alg in (Tsit5(), Rosenbrock23()), (cb, reference) in (
                (ProbIntsUncertainty(0.1, 2), _reference_probints(0.1, 2)),
                (AdaptiveProbIntsUncertainty(2), _reference_adaptive_probints(2)),
            )
        prob = ODEProblem(g, [1.0, 0.0, 0.0], (0.0, 1.0), [10.0, 28.0, 8 / 3])
        Random.seed!(1234)
        actual = solve(prob, alg; callback = cb)
        Random.seed!(1234)
        expected = solve(prob, alg; callback = reference)
        @test actual.retcode == expected.retcode == ReturnCode.Success
        @test reinterpret(UInt64, actual.t) == reinterpret(UInt64, expected.t)
        @test length(actual.u) == length(expected.u)
        @test all(reinterpret(UInt64, u) == reinterpret(UInt64, v) for (u, v) in zip(actual.u, expected.u))
        @test _probints_affect_bytes(alg, cb) == 0
    end
end

function _probints_decay!(du, u, p, t)
    du .= -p .* u
    return nothing
end

@testset "ProbInts operator noise scales" begin
    for alg in (Tsit5(), Rosenbrock23()), u0 in (ones(8), ones(2, 2))
        n = size(u0, 1)
        for σ in (
                0.2I, Diagonal(range(0.1, 0.2; length = n)),
                0.1Matrix{Float64}(I, n, n) .+ 0.1 / n,
            )
            @testset "$(typeof(alg)), $(size(u0)), $(typeof(σ))" begin
                prob = ODEProblem(_probints_decay!, u0, (0.0, 1.0), 0.7)
                Random.seed!(1234)
                expected = solve(prob, alg; callback = _reference_probints(σ, 2))
                @test expected.retcode == ReturnCode.Success
                Random.seed!(1234)
                actual = solve(prob, alg; callback = ProbIntsUncertainty(σ, 2))
                @test actual.retcode == ReturnCode.Success
                @test isequal(actual.t, expected.t)
                @test isequal(actual.u, expected.u)
            end
        end
    end
end

@testset "Seeded ProbInts out-of-place solutions" begin
    for alg in (Tsit5(), Rosenbrock23()), T in (Float64, ComplexF64), (cb, reference) in (
                (ProbIntsUncertainty(0.1, 2), _reference_probints(0.1, 2)),
                (AdaptiveProbIntsUncertainty(2), _reference_adaptive_probints(2)),
            )
        f = ODEFunction(
            (u, p, t) -> -p .* u;
            jac = (u, p, t) -> -p * Matrix{T}(I, length(u), length(u)),
            tgrad = (u, p, t) -> zero(u)
        )
        prob = ODEProblem(f, ones(T, 8), (0.0, 1.0), 0.7)
        Random.seed!(1234)
        actual = solve(prob, alg; callback = cb)
        Random.seed!(1234)
        expected = solve(prob, alg; callback = reference)
        @test actual.retcode == expected.retcode == ReturnCode.Success
        @test isequal(actual.t, expected.t)
        @test isequal(actual.u, expected.u)
    end
end

function _probints_endpoint(p)
    Random.seed!(1234)
    prob = ODEProblem(_probints_decay!, fill(one(p), 8), (0.0, 1.0), p)
    sol = solve(
        prob, Euler(); dt = 0.1, adaptive = false, callback = ProbIntsUncertainty(0.2, 1, false),
        save_everystep = false
    )
    return sol.u[end][1]
end

@testset "ProbInts parameter derivative" begin
    Random.seed!(1234)
    u, derivative = ones(8), zeros(8)
    for _ in 1:10
        derivative .= (1 - 0.1 * 0.7) .* derivative .- 0.1 .* u
        u .= (1 - 0.1 * 0.7) .* u .+ 0.2 * sqrt(0.1^2) * randn(8)
    end
    actual = ForwardDiff.derivative(_probints_endpoint, 0.7)
    finite_difference = (_probints_endpoint(0.7 + 1.0e-5) - _probints_endpoint(0.7 - 1.0e-5)) / 2.0e-5
    @test _probints_endpoint(0.7) ≈ u[1] atol = 1.0e-14
    @test actual ≈ derivative[1] atol = 1.0e-12
    @test actual ≈ finite_difference atol = 1.0e-9
end

function _probints_wrapper_endpoint(make_u0, p)
    Random.seed!(1234)
    prob = ODEProblem(_probints_decay!, make_u0(p), (0.0, 1.0), p)
    sol = solve(
        prob, Euler(); dt = 0.125, adaptive = false,
        callback = ProbIntsUncertainty(0.2, 1, false), save_everystep = false
    )
    return sol.u[end][1]
end

@testset "ProbInts wrapper-state derivative" begin
    Random.seed!(1234)
    u, derivative = ones(8), zeros(8)
    for _ in 1:8
        derivative .= (1 - 0.125 * 0.7) .* derivative .- 0.125 .* u
        u .= (1 - 0.125 * 0.7) .* u .+ 0.2 * sqrt(0.125^2) * randn(8)
    end
    for make_u0 in (
            p -> ArrayPartition(fill(one(p), 4), fill(one(p), 4)),
            p -> MVector{8}(fill(one(p), 8)),
        )
        endpoint = Base.Fix1(_probints_wrapper_endpoint, make_u0)
        actual = ForwardDiff.derivative(endpoint, 0.7)
        finite_difference = (endpoint(0.7 + 1.0e-5) - endpoint(0.7 - 1.0e-5)) / 2.0e-5
        @test endpoint(0.7) ≈ u[1] atol = 1.0e-14
        @test actual ≈ derivative[1] atol = 1.0e-12
        @test actual ≈ finite_difference atol = 1.0e-9
    end
end

@testset "ProbInts non-Float64 noise" begin
    for T in (Float32, BigFloat, ComplexF64), (cb, reference) in (
                (ProbIntsUncertainty(0.1, 2), _reference_probints(0.1, 2)),
                (AdaptiveProbIntsUncertainty(2), _reference_adaptive_probints(2)),
            )
        prob = ODEProblem(_probints_decay!, ones(T, 8), (0.0, 1.0), 0.7)
        Random.seed!(1234)
        actual = solve(prob, Tsit5(); callback = cb)
        Random.seed!(1234)
        expected = solve(prob, Tsit5(); callback = reference)
        @test actual.retcode == expected.retcode == ReturnCode.Success
        @test isequal(actual.t, expected.t)
        @test isequal(actual.u, expected.u)
    end
end

# Out-of-place Vector problems have no tmp cache; must not MethodError.
lor_oop(u, p, t) = [
    10 * (u[2] - u[1]), u[1] * (28 - u[3]) - u[2], u[1] * u[2] - (8 / 3) * u[3],
]
p_oop = ODEProblem(lor_oop, [1.0, 0.0, 0.0], (0.0, 1.0))
@test SciMLBase.successful_retcode(
    solve(p_oop, Tsit5(); callback = ProbIntsUncertainty(0.1, 5))
)
@test SciMLBase.successful_retcode(
    solve(p_oop, Tsit5(); callback = AdaptiveProbIntsUncertainty(5))
)

# Complex states receive real Float64 noise.
function _lin!(du, u, p, t)
    du .= -0.5 .* u
    return nothing
end
Random.seed!(1234)
sol_c = solve(
    ODEProblem(_lin!, ones(ComplexF64, 3), (0.0, 2.0)), Tsit5();
    callback = ProbIntsUncertainty(0.1, 5)
)
@test SciMLBase.successful_retcode(sol_c)
@test all(iszero, imag.(sol_c.u[end]))
Random.seed!(1234)
sol_c2 = solve(
    ODEProblem(_lin!, ones(ComplexF64, 3), (0.0, 2.0)), Tsit5();
    callback = ProbIntsUncertainty(0.1, 5)
)
@test sol_c.u[end] == sol_c2.u[end]
