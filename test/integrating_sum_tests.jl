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

# integrand_inplace = true: in-place integrand with an out-of-place problem
# (e.g. immutable state with a mutable, parameter-shaped integrand buffer)
using StaticArrays
prob_oop = ODEProblem((u, p, t) -> SVector(1.0), SVector(0.0), (0.0, 1.0))
integrated = IntegrandValuesSum(zeros(1))
sol = solve(
    prob_oop, Euler(),
    callback = IntegratingSumCallback(
        (out, u, t, integrator) -> (out[1] = u[1]; nothing), integrated, Float64[0.0];
        integrand_inplace = true
    ),
    dt = 0.1
)
@test integrated.integrand[1] == 0.5

# integrand_inplace = true requires a prototype
@test_throws ArgumentError IntegratingSumCallback(
    (out, u, t, integrator) -> nothing, IntegrandValuesSum(zeros(1)), nothing;
    integrand_inplace = true
)

# integrand_inplace = false forces the allocating form even for in-place problems
prob_iip = ODEProblem((du, u, p, t) -> (du[1] = 1.0; nothing), [0.0], (0.0, 1.0))
integrated = IntegrandValuesSum(zeros(1))
sol = solve(
    prob_iip, Euler(),
    callback = IntegratingSumCallback(
        (u, t, integrator) -> [u[1]], integrated, Float64[0.0];
        integrand_inplace = false
    ),
    dt = 0.1
)
@test integrated.integrand[1] == 0.5
