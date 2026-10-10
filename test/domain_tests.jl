using DiffEqCallbacks, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5,
    OrdinaryDiffEqRosenbrock, Test, ADTypes, NonlinearSolve, StaticArrays, SciMLBase

# Non-negative ODE examples
#
# Reference:
# Shampine, L.F., S. Thompson, J.A. Kierzenka, and G.D. Byrne,
# "Non-negative solutions of ODEs," Applied Mathematics and Computation Vol. 170, 2005,
# pp. 556-569.
# https://www.mathworks.com/help/matlab/math/nonnegative-ode-solution.html

"""
Absolute value function

```math
\\frac{du}{dt} = -|u|
```

with initial condition ``u₀=1``, and solution

```math
u(t) = u₀*e^{-t}
```

for positive initial values ``u₀``.
"""
function absval(du, u, p, t)
    return du[1] = -abs(u[1])
end
analytic(u₀, p, t) = u₀ * exp(-t)
ff = ODEFunction(absval, analytic = analytic)
prob_absval = ODEProblem(ff, [1.0], (0.0, 40.0))

# naive approach leads to large errors
naive_sol_absval = solve(prob_absval, BS3())
@test naive_sol_absval.errors[:l∞] > 9.0e4
@test naive_sol_absval.errors[:l2] > 1.3e4

# general domain approach
function g_domain(resid, u, p)
    return resid[1] = u[1] < 0 ? -u[1] : 0
end
general_sol_absval = solve(
    prob_absval, BS3();
    callback = GeneralDomain(
        g_domain, [1.0];
        autodiff = AutoForwardDiff(),
        nlsolve = NewtonRaphson(; autodiff = AutoForwardDiff())
    ),
    save_everystep = false
)
@test all(x -> x[1] ≥ 0, general_sol_absval.u)
@test general_sol_absval.errors[:l∞] < 9.9e-5
@test general_sol_absval.errors[:l2] < 4.5e-5
@test general_sol_absval.errors[:final] < 4.3e-18

# test "non-autonomous" function
g_domain_t(resid, u, p, t) = g_domain(resid, u, p)

general_t_sol_absval = solve(
    prob_absval, BS3();
    callback = GeneralDomain(
        g_domain_t, [1.0];
        autodiff = AutoForwardDiff(),
        nlsolve = NewtonRaphson(; autodiff = AutoForwardDiff())
    ),
    save_everystep = false
)
@test general_sol_absval.t ≈ general_t_sol_absval.t
@test general_sol_absval.u ≈ general_t_sol_absval.u

# positive domain approach
positive_sol_absval = solve(
    prob_absval, BS3(); callback = PositiveDomain([1.0]),
    save_everystep = false
)
@test all(x -> x[1] ≥ 0, positive_sol_absval.u)
@test general_sol_absval.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]

mutable struct StaticStateIntegrator{U}
    u::U
end

static_u = @SVector [-1.0, 2.0]
static_integrator = StaticStateIntegrator(static_u)
@test DiffEqCallbacks._set_neg_zero!(static_integrator, static_u)
@test static_integrator.u == @SVector [0.0, 2.0]
@test static_integrator.u isa typeof(static_u)

# specify abstol as array or scalar
positive_sol_absval2 = solve(
    prob_absval, BS3();
    callback = PositiveDomain([1.0], abstol = [1.0e-9]),
    save_everystep = false
)
@test all(x -> x[1] ≥ 0, positive_sol_absval2.u)
@test positive_sol_absval2.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]
positive_sol_absval3 = solve(
    prob_absval, BS3();
    callback = PositiveDomain([1.0], abstol = 1.0e-9),
    save_everystep = false
)
@test all(x -> x[1] ≥ 0, positive_sol_absval3.u)
@test positive_sol_absval3.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]

# specify scalefactor
positive_sol_absval4 = solve(
    prob_absval, BS3();
    callback = PositiveDomain([1.0], scalefactor = 0.2),
    save_everystep = false
)
@test all(x -> x[1] ≥ 0, positive_sol_absval4.u)
@test positive_sol_absval4.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]

"""
Knee problem

```math
\\frac{du}{dt} = \epsilon^{-1}(1-t-u)u
```

with initial condition ``u0=1``, and generally ``0 < \epsilon << 1``.
Here ``\epsilon=1e-6``. Then the solution approaches ``u=1-t`` for ``t<1``
and ``u=0`` for ``t>1``.
"""
function knee(du, u, p, t)
    return du[1] = 1.0e6 * (1 - t - u[1]) * u[1]
end

prob_knee = ODEProblem(knee, [1.0], (0.0, 2.0))

# unfortunately callbacks do not work with solver CVODE_BDF which is comparable to ode15s
# used in MATLAB example, so we use Rodas5
naive_sol_knee = solve(prob_knee, Rodas5P())
@test naive_sol_knee[1, end] ≈ -1.0 atol = 1.0e-5

# positive domain approach
positive_sol_knee = solve(
    prob_knee, Rodas5P(); callback = PositiveDomain([1.0]),
    save_everystep = false
)
@test all(x -> x[1] ≥ 0, positive_sol_knee.u)
@test positive_sol_knee[1, end] ≈ 0.0 atol = 1.0e-5

## Now test on out-of-place equations
r, K = 1.1, 10.0
logistic(u, p, t) = u * r * (1 - u / K)
t = (0.0, 20.0)
logistic_p = ODEProblem(logistic, 0.02, t)
logistic_s = solve(logistic_p, Tsit5())
logistic_s_positive = solve(logistic_p, Tsit5(), callback = PositiveDomain())

# Out-of-place Vector: get_tmp_cache is nothing; default PositiveDomain() must still work.
absval_oop(u, p, t) = -abs.(u)
prob_absval_oop = ODEProblem(absval_oop, [1.0], (0.0, 40.0))
positive_sol_absval_oop = solve(
    prob_absval_oop, Tsit5(); callback = PositiveDomain(), save_everystep = false
)
@test all(x -> x[1] ≥ 0, positive_sol_absval_oop.u)
@test SciMLBase.successful_retcode(positive_sol_absval_oop)

function _posdom_decay!(du, u, p, t)
    du[1] = -0.001 * u[1] + u[2] - (-0.001 * 2.0 + 2.0)
    du[2] = -u[1] - 0.001 * u[2] - (-2.0 - 0.001 * 2.0)
    du[3] = -0.001 * u[3] + 2.0 * u[4] - (-0.001 * 2.0 + 2.0 * 2.0)
    du[4] = -2.0 * u[3] - 0.001 * u[4] - (-2.0 * 2.0 - 0.001 * 2.0)
    return nothing
end
function _positivedomain_bytes_per_step(n)
    u0 = [3.0, 2.0, 2.5, 2.0]
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(_posdom_decay!, u0, (0.0, 1.0e4))
    integ = init(prob, Tsit5(); callback = PositiveDomain(), save_everystep = false)
    for _ in 1:20
        step!(integ)
    end
    step!(integ)
    return (
        @allocated for _ in 1:n
            step!(integ)
        end
    ) / n
end
@test _positivedomain_bytes_per_step(200) < 230.0

# Bitwise match default vs user buffer on a problem that clips negatives.
let
    sol_default = solve(
        prob_absval, BS3();
        callback = PositiveDomain(; save = false)
    )
    sol_buf = solve(
        prob_absval, BS3();
        callback = PositiveDomain([1.0]; save = false)
    )
    @test any(x -> x[1] == 0, sol_default.u)
    @test length(sol_default.t) > 2
    @test sol_default.t == sol_buf.t
    @test sol_default.u == sol_buf.u
end
