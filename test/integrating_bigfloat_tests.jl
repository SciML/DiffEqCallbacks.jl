using OrdinaryDiffEqLowOrderRK, DiffEqCallbacks, Functors
using Test

# u' = 1 gives u = t exactly, so the integral of u^2 over [0, 1] is exactly
# 1/3. RK4's interpolation is exact for this solution and its 2-point Gauss
# rule is exact for quadratics, so the observed error is set by the
# quadrature nodes and weights alone: Float64 tables cap it near 1e-17 while
# tables converted to the integrand's element type track the state
# precision (BigFloat error below 1e-70).
for T in (Float64, BigFloat)
    prob = ODEProblem((u, p, t) -> one(u), zero(T), (zero(T), one(T)))
    exact = one(T) / 3
    tight = T === BigFloat ? big"1e-70" : 1.0e-13

    integrated = IntegrandValuesSum(zero(T))
    sol = solve(
        prob, RK4(),
        callback = IntegratingSumCallback(
            (u, t, integrator) -> u^2, integrated, zero(T)
        ),
        dt = T(0.1)
    )
    @test abs(integrated.integrand - exact) < tight

    integrated_gk = IntegrandValuesSum(zero(T))
    gk_tol = T === BigFloat ? big"1e-40" : 1.0e-14
    sol = solve(
        prob, RK4(),
        callback = IntegratingGKSumCallback(
            (u, t, integrator) -> u^2, integrated_gk, zero(T), gk_tol
        ),
        dt = T(0.1)
    )
    @test abs(integrated_gk.integrand - exact) < tight

    values = IntegrandValues(T, T)
    sol = solve(
        prob, RK4(),
        callback = IntegratingCallback((u, t, integrator) -> u^2, values, zero(T)),
        dt = T(0.1)
    )
    @test abs(sum(values.integrand) - exact) < tight

    values_gk = IntegrandValues(T, T)
    sol = solve(
        prob, RK4(),
        callback = IntegratingGKCallback(
            (u, t, integrator) -> u^2, values_gk, zero(T), gk_tol
        ),
        dt = T(0.1)
    )
    @test abs(sum(values_gk.integrand) - exact) < tight
end

# Vector-valued BigFloat state
prob = ODEProblem((u, p, t) -> one.(u), [big(0.0)], (big(0.0), big(1.0)))
integrated = IntegrandValuesSum([big(0.0)])
sol = solve(
    prob, RK4(),
    callback = IntegratingSumCallback(
        (u, t, integrator) -> u .^ 2, integrated, [big(0.0)]
    ),
    dt = big(0.1)
)
@test abs(integrated.integrand[1] - big(1) / 3) < 1.0e-70

# Gauss-Kronrod tolerances accept any real type, including BigFloat.
@test_nowarn IntegratingGKSumCallback(
    (u, t, integrator) -> u^2, IntegrandValuesSum(big(0.0)), big(0.0), big"1e-30"
)
@test_nowarn IntegratingGKCallback(
    (u, t, integrator) -> u^2, IntegrandValues(BigFloat, BigFloat), big(0.0), big"1e-30"
)

# High-precision string tables agree with the Float64 literals, and the
# converted BigFloat weights sum to the interval length.
for (strs, lit) in [
        (
            DiffEqCallbacks._GAUSS_POINTS_STR, DiffEqCallbacks.gauss_points,
        ),
        (
            DiffEqCallbacks._GAUSS_WEIGHTS_STR, DiffEqCallbacks.gauss_weights,
        ),
        (DiffEqCallbacks._GK_POINTS_STR, DiffEqCallbacks.gk_points),
        (DiffEqCallbacks._GK_WEIGHTS_STR, DiffEqCallbacks.gk_weights),
        (DiffEqCallbacks._G_WEIGHTS_STR, DiffEqCallbacks.g_weights),
    ]
    for n in 1:10
        @test maximum(abs.(parse.(Float64, strs[n]) .- lit[n])) < 1.0e-14
    end
end
for n in 1:10
    _, gauss_ws = DiffEqCallbacks._gauss_tables(BigFloat)
    gk_pts, gk_ws, _ = DiffEqCallbacks._gk_tables(BigFloat)
    @test abs(sum(gauss_ws[n]) - 2) < big"1e-70"
    @test abs(sum(gk_ws[n]) - 2) < big"1e-70"
    @test isapprox(sum(gk_pts[n]), big(0.0); atol = big"1e-70")
end

@testset "quadrature eltype handles NamedTuple and Any" begin
    # Direct unit tests for _quadrature_eltype with Any and nested containers
    @test DiffEqCallbacks._quadrature_eltype(Any, Float64, Any) === Float64
    @test DiffEqCallbacks._quadrature_eltype(Any, Any) === Float64
    let p_nt = (; x = (; αβ = [1.5, 1.0]), δγ = [3.0, 1.0])
        @test eltype(p_nt) === Any
        @test DiffEqCallbacks._quadrature_leaf_eltype(typeof(p_nt)) === Float64
        values_nt = IntegrandValues(Float64, typeof(p_nt))
        integrand_el = eltype(eltype(values_nt.integrand))
        time_el = eltype(values_nt.ts)
        prototype_el = eltype(p_nt)
        @test DiffEqCallbacks._quadrature_eltype(integrand_el, time_el, prototype_el) ===
            Float64
        @test DiffEqCallbacks._quadrature_eltype(integrand_el, prototype_el) === Float64
    end
    let p_nt_big = (; a = [big(1.0)], b = [big(2.0)])
        @test DiffEqCallbacks._quadrature_leaf_eltype(typeof(p_nt_big)) === BigFloat
        @test DiffEqCallbacks._quadrature_eltype(BigFloat, Float64, BigFloat) === BigFloat
    end
    # Mixed nested types promote to higher precision
    @test DiffEqCallbacks._quadrature_leaf_eltype(typeof((; u = [1.0], v = [big(1.0)]))) === BigFloat
end

@testset "IntegratingCallback accepts NamedTuple prototype" begin
    p_nt = (; x = (; αβ = [1.5, 1.0]), δγ = [3.0, 1.0])
    for T in ("IntegratingCallback", "IntegratingGKCallback")
        if T == "IntegratingCallback"
            vals = IntegrandValues(Float64, typeof(p_nt))
            cb = IntegratingCallback((u, t, integrator) -> p_nt, vals, p_nt)
        else
            vals = IntegrandValues(Float64, typeof(p_nt))
            cb = IntegratingGKCallback((u, t, integrator) -> p_nt, vals, p_nt)
        end
        @test cb !== nothing
    end
    vals_sum = IntegrandValuesSum(p_nt)
    @test_nowarn IntegratingSumCallback(((u, t, integrator) -> p_nt), vals_sum, p_nt)
    @test_nowarn IntegratingGKSumCallback(((u, t, integrator) -> p_nt), vals_sum, p_nt)
end
