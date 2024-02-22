using Test, OrdinaryDiffEq, DiffEqCallbacks, LinearAlgebra,
      SciMLSensitivity, Tracker
import LinearAlgebra: norm
import ODEProblemLibrary: prob_ode_2Dlinear,
                          prob_ode_linear, prob_ode_vanderpol, prob_ode_rigidbody,
                          prob_ode_nonlinchem, prob_ode_lorenz
using DiffEqCallbacks: sample, store!

# save_everystep, scalar problem
prob = prob_ode_linear
saved_values = SavedValues(Float64, Float64)
cb = SavingCallback((u, t, integrator) -> u, saved_values)
sol = solve(prob, Tsit5(), callback = cb)
print("\n", saved_values, "\n")
@test all(idx -> sol.t[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> sol.u[idx] == saved_values.saveval[idx], eachindex(saved_values.t))

# save_everystep, inplace problem
prob2D = prob_ode_2Dlinear
saved_values = SavedValues(eltype(prob2D.tspan), typeof(prob2D.u0))
cb = SavingCallback((u, t, integrator) -> copy(u), saved_values)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> sol.t[idx] .== saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> all(sol.u[idx] .== saved_values.saveval[idx]), eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D.tspan), eltype(prob2D.u0))
cb = SavingCallback((u, t, integrator) -> u[1], saved_values)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> sol.t[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> sol.u[idx][1] == saved_values.saveval[idx], eachindex(saved_values.t))

# saveat, scalar problem
saved_values = SavedValues(Float64, Float64)
saveat = range(prob.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u, saved_values, saveat = saveat)
sol = solve(prob, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

# scalar saveat, scalar problem
saved_values = SavedValues(Float64, Float64)
saveat = range(prob.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u, saved_values, saveat = step(saveat))
sol = solve(prob, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

# scalar saveat without start and end
saved_values = SavedValues(Float64, Float64)
cb = SavingCallback((u, t, integrator) -> u,
    saved_values;
    saveat = 0.2,
    save_start = false,
    save_end = false)
preset = PresetTimeCallback(Float64[], identity)
sol = solve(prob, Tsit5(); dt = 0.2, adaptive = false, callback = CallbackSet(cb, preset))
@test sol.t ≈ 0.0:0.2:1.0
@test saved_values.t ≈ 0.2:0.2:0.8

# saveat, inplace problem
saved_values = SavedValues(eltype(prob2D.tspan), typeof(prob2D.u0))
saveat = range(prob2D.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> copy(u), saved_values, saveat = saveat)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> norm(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D.tspan), eltype(prob2D.u0))
saveat = range(prob2D.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u[1], saved_values, saveat = saveat)
sol = solve(prob2D, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx])[1] - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

# saveat, tdir<0, scalar problem
prob_inverse = ODEProblem(prob.f, prob.u0, (prob.tspan[end], prob.tspan[1]), 1.01)
saved_values = SavedValues(Float64, Float64)
saveat = range(prob_inverse.tspan[1], stop = prob.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u, saved_values, saveat = saveat, tdir = -1)
sol = solve(prob_inverse, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

# saveat, tdir<0, inplace problem
prob2D_inverse = ODEProblem(prob2D.f, prob2D.u0, (prob2D.tspan[end], prob2D.tspan[1]), 1.01)
saved_values = SavedValues(eltype(prob2D_inverse.tspan), typeof(prob2D_inverse.u0))
saveat = range(prob2D_inverse.tspan[1], stop = prob2D_inverse.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> copy(u), saved_values, saveat = saveat, tdir = -1)
sol = solve(prob2D_inverse, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> norm(sol(saveat[idx]) - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

saved_values = SavedValues(eltype(prob2D_inverse.tspan), eltype(prob2D_inverse.u0))
saveat = range(prob2D_inverse.tspan[1], stop = prob2D_inverse.tspan[2], length = 50)
cb = SavingCallback((u, t, integrator) -> u[1], saved_values, saveat = saveat, tdir = -1)
sol = solve(prob2D_inverse, Tsit5(), callback = cb)
@test all(idx -> saveat[idx] == saved_values.t[idx], eachindex(saved_values.t))
@test all(idx -> abs(sol(saveat[idx])[1] - saved_values.saveval[idx]) < 8.e-15,
    eachindex(saved_values.t))

# Make sure it doesn't error with mutable in oop
prob = ODEProblem((u, p, t) -> u, rand(4, 4), (0.0, 1.0))
saved_values = SavedValues(Float64, Tuple{Float64, Float64})
cb = SavingCallback((u, t, integrator) -> (tr(u), norm(u)), saved_values,
    saveat = 0.0:0.1:1.0)
sol = solve(prob, Tsit5(), callback = cb)

# Save only end
prob = ODEProblem((du, u, p, t) -> du .= u, rand(4, 4), (0.0, 1.0))
saved_values = SavedValues(Float64, Tuple{Float64, Float64})
cb = SavingCallback((u, t, integrator) -> (tr(u), norm(u)), saved_values,
    save_everystep = false, save_start = false)
sol = solve(prob, Tsit5(), callback = cb)
@test length(saved_values.t) == 1
@test saved_values.t[1] == 1.0

# Tracker with Saving Callback
## This is pretty much a hack. It has been merged into DistributionsAD master
Base.prevfloat(r::Tracker.TrackedReal) = Tracker.track(prevfloat, r)
Tracker.@grad function prevfloat(r::Real)
    prevfloat(Tracker.data(r)), Δ -> (Δ,)
end
Base.nextfloat(r::Tracker.TrackedReal) = Tracker.track(nextfloat, r)
Tracker.@grad function nextfloat(r::Real)
    nextfloat(Tracker.data(r)), Δ -> (Δ,)
end

function rober(u, p::TrackedArray, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return Tracker.collect([-k₁ * y₁ + k₃ * y₂ * y₃,
        k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃,
        k₂ * y₂^2])
end

p = TrackedArray([1.9f0, 1.0f0, 3.0f0])
u0 = TrackedArray([1.0f0, 0.0f0, 0.0f0])
tspan = TrackedArray([0.0f0, 1.0f0])
prob = ODEProblem{false}(rober, u0, tspan, p)
saved_values = SavedValues(eltype(tspan), eltype(p))
cb = SavingCallback((u, t, integrator) -> integrator.EEst * integrator.dt, saved_values)

@test !all(iszero.(Tracker.gradient(
    p -> begin
        solve(remake(prob, u0 = u0, p = p, tspan = tspan),
            Tsit5(),
            sensealg = SensitivityADPassThrough(),
            callback = cb)
        return sum(saved_values.saveval)
    end,
    p)[1]))

# Test that our `LinearizingSavingCallback` gives back something that when interpolated,
# respects our `abstol`/`reltol` versus the actual solution:
using DataInterpolations
import DiffEqCallbacks: as_array, finish!

as_array(T::Type{<:AbstractArray}) = T
as_array(T::Type{<:Number}) = Vector{T}

if VERSION >= v"1.9" # stack
    function test_linearization(prob,
            solver;
            max_deriv = 0,
            abstol = 1e-6,
            reltol = 1e-3)

        # Solve the given problem once, saving the primal and some number of derivatives
        ils = IndependentlyLinearizedSolution(prob, max_deriv)
        lsc = LinearizingSavingCallback(ils)
        sol = solve(prob,
            solver;
            callback = lsc,
            abstol,
            reltol)
        @test sol.retcode == ReturnCode.Success
        @test ils.ilsc === nothing

        N = length(ils)
        t_upsampled = LinearInterpolation(ils.ts, Float64.(1:N))(range(1,
            N;
            length = 10 * N))
        for deriv_idx in 0:max_deriv
            u_linear_upsampled = sample(ils, t_upsampled, deriv_idx)
            u_interp_upsampled = stack(as_array.(sol(
                t_upsampled, Val{deriv_idx}; continuity = :left).u))'

            check = isapprox(u_linear_upsampled,
                u_interp_upsampled;
                # We loosen the comparison bounds here as higher derivative orders
                # have worse accuracy guarantees, approximately `sqrt()` for each order.
                atol = abstol^(2.0^(-deriv_idx)),
                rtol = reltol^(2.0^(-deriv_idx)))
            if !check
                @error("Check failed", solver, deriv_idx)
                display(abs.(u_linear_upsampled .- u_interp_upsampled))
            end
            @test check
        end
    end

    max_deriv_map = Dict(
        Tsit5 => 2,
        Rodas5P => 2,
        Rosenbrock23 => 1
    )
    for solver in [Tsit5, Rodas5P, Rosenbrock23]
        max_deriv = max_deriv_map[solver]
        @testset "$(solver)" begin
            test_linearization(prob_ode_linear, solver(); max_deriv)
            test_linearization(
                prob_ode_linear, solver(); abstol = 1e-9, reltol = 1e-9, max_deriv)
            test_linearization(prob_ode_vanderpol, solver(); max_deriv)
            test_linearization(prob_ode_rigidbody, solver(); max_deriv)
            test_linearization(prob_ode_nonlinchem, solver(); max_deriv)
            test_linearization(prob_ode_lorenz, solver(); max_deriv)
        end
    end

    @testset "fail gracefully" begin
        f_error2(du, u, p, t) = du .= u ./ t .- 1
        u0 = [1.0]
        du0 = [1.0]
        prob = DAEProblem(f_error2, u0, du0, (0.0, 1.0); differential_vars = [true])
        ils = IndependentlyLinearizedSolution(prob, 0)
        lsc = LinearizingSavingCallback(ils)
        sol = solve(prob, DFBDF(); callback = lsc)  # this would if we were not failing with grace
        @test sol.retcode == ReturnCode.InitialFailure
        @test isempty(ils.ts)
        @test isempty(ils.us)
        @test isempty(ils.time_mask)
    end

    # We do not support 2d states yet.
    #test_linearization(prob_ode_2Dlinear, Tsit5())
end
