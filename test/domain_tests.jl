using DiffEqCallbacks, OrdinaryDiffEq, Test

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
function absval(du,u,p,t)
    du[1] = -abs(u[1])
end
analytic(u₀, p, t) = u₀*exp(-t)
ff = ODEFunction(absval,analytic=analytic)
prob_absval = ODEProblem(ff, [1.0], (0.0, 40.0))

# naive approach leads to large errors
naive_sol_absval = solve(prob_absval, BS3())
@test naive_sol_absval.errors[:l∞] > 9e4
@test naive_sol_absval.errors[:l2] > 1.3e4

# general domain approach
function g(resid,u)
    resid[1] = u[1] < 0 ? -u[1] : 0
end
general_sol_absval = solve(prob_absval, BS3(); callback=GeneralDomain(g, [1.0]), save_everystep=false)
@test all(x -> x[1] ≥ 0, general_sol_absval.u)
@test general_sol_absval.errors[:l∞] < 9.9e-5
@test general_sol_absval.errors[:l2] < 4.5e-5
@test general_sol_absval.errors[:final] < 4.3e-18

# test non-autonomous function
g_t(resid, u, p, t) = g(resid, u)

general_t_sol_absval = solve(prob_absval, BS3(); callback=GeneralDomain(g_t, [1.0]), save_everystep=false)
@test general_sol_absval.t ≈ general_t_sol_absval.t
@test general_sol_absval.u ≈ general_t_sol_absval.u

# positive domain approach
positive_sol_absval = solve(prob_absval, BS3(); callback=PositiveDomain([1.0]), save_everystep=false)
@test all(x -> x[1] ≥ 0, positive_sol_absval.u)
@test general_sol_absval.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]

# specify abstol as array or scalar
positive_sol_absval2 = solve(prob_absval, BS3(); callback=PositiveDomain([1.0], abstol=[1e-9]), save_everystep=false)
@test all(x -> x[1] ≥ 0, positive_sol_absval2.u)
@test positive_sol_absval2.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]
positive_sol_absval3 = solve(prob_absval, BS3(); callback=PositiveDomain([1.0], abstol=1e-9), save_everystep=false)
@test all(x -> x[1] ≥ 0, positive_sol_absval3.u)
@test positive_sol_absval3.errors[:l∞] ≈ positive_sol_absval.errors[:l∞]

# specify scalefactor
positive_sol_absval4 = solve(prob_absval, BS3(); callback=PositiveDomain([1.0], scalefactor=0.2), save_everystep=false)
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
function knee(du,u,p,t)
    du[1] = 1e6*(1-t-u[1])*u[1]
end

prob_knee = ODEProblem(knee, [1.0], (0.0, 2.0))

# unfortunately callbacks do not work with solver CVODE_BDF which is comparable to ode15s
# used in MATLAB example, so we use Rodas5
naive_sol_knee = solve(prob_knee, Rodas5())
@test naive_sol_knee[1, end] ≈ -1.0 atol=1e-5

# positive domain approach
positive_sol_knee = solve(prob_knee, Rodas5(); callback=PositiveDomain([1.0]), save_everystep=false)
@test all(x -> x[1] ≥ 0, positive_sol_knee.u)
@test positive_sol_knee[1, end] ≈ 0.0 atol=1e-5

## Now test on out-of-place equations
r, K = 1.1, 10.0
logistic(u,p,t) = u*r*(1-u/K)
t = (0.0, 20.0)
logistic_p = ODEProblem(logistic, 0.02, t)
logistic_s = solve(logistic_p, Tsit5())
logistic_s_positive = solve(logistic_p, Tsit5(), callback=PositiveDomain())
