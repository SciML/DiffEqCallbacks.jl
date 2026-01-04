using OrdinaryDiffEqTsit5, DiffEqCallbacks, DiffEqBase
using SciMLSensitivity, Tracker
using Test

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
    return Tracker.collect(
        [
            -k₁ * y₁ + k₃ * y₂ * y₃,
            k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃,
            k₂ * y₂^2,
        ]
    )
end

p = TrackedArray([1.9f0, 1.0f0, 3.0f0])
u0 = TrackedArray([1.0f0, 0.0f0, 0.0f0])
tspan = TrackedArray([0.0f0, 1.0f0])
prob = ODEProblem{false}(rober, u0, tspan, p)
saved_values = SavedValues(eltype(tspan), eltype(p))
cb = SavingCallback((u, t, integrator) -> integrator.EEst * integrator.dt, saved_values)

@test !all(
    iszero.(
        Tracker.gradient(
            p -> begin
                solve(
                    remake(prob, u0 = u0, p = p, tspan = tspan),
                    Tsit5(),
                    sensealg = SensitivityADPassThrough(),
                    callback = cb
                )
                return sum(saved_values.saveval)
            end,
            p
        )[1]
    )
)
