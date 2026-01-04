using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5, DiffEqCallbacks
using Test

prob = ODEProblem((u, p, t) -> [1.0], [0.0], (0.0, 1.0))
integrated = IntegrandValuesSum(zeros(1))
sol = solve(
    prob, Euler(),
    callback = IntegratingSumCallback(
        (u, t, integrator) -> [1.0], integrated, Float64[0.0]
    ),
    dt = 0.1
)
@test integrated.integrand[1] == 1
integrated = IntegrandValuesSum(zeros(1))
sol = solve(
    prob, Euler(),
    callback = IntegratingSumCallback(
        (u, t, integrator) -> [u[1]], integrated, Float64[0.0]
    ),
    dt = 0.1
)
@test integrated.integrand[1] == 0.5
