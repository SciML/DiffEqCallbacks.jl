using DiffEqCallbacks, OrdinaryDiffEq, Base.Test

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
function absval(t,u,du)
    du[1] = -abs(u[1])
end
(f::typeof(absval))(::Type{Val{:analytic}}, t, u₀) = u₀*exp(-t)
prob_absval = ODEProblem(absval, [1.0], (0.0, 40.0))

# naive approach leads to large errors
naive_sol_absval = solve(prob_absval, BS3())
@test naive_sol_absval.errors[:l∞] > 9e4
@test naive_sol_absval.errors[:l2] > 1.3e4

# general domain approach
# can only guarantee approximately non-negative values
function g(u,resid)
    resid[1] = u[1] < 0 ? -u[1] : 0
end

general_sol_absval = solve(prob_absval, BS3(); callback=GeneralDomain(g, [1.0]))
@test all(x -> x[1] ≥ -10*eps(), general_sol_absval.u)
@test general_sol_absval.errors[:l∞] < 9.9e-5
@test general_sol_absval.errors[:l2] < 4.3e-5
@test general_sol_absval.errors[:final] < 4.4e-18

# test non-autonomous function
g_t(t, u, resid) = g(u, resid)

general_t_sol_absval = solve(prob_absval, BS3(); callback=GeneralDomain(g_t, [1.0]))
@test general_sol_absval.t == general_t_sol_absval.t &&
    general_sol_absval.u == general_t_sol_absval.u

# positive domain approach
# can guarantee non-negative values
positive_sol_absval = solve(prob_absval, BS3(); callback=PositiveDomain([1.0]))
@test all(x -> x[1] ≥ 0, positive_sol_absval.u)
@test positive_sol_absval.errors[:l∞] < 9.9e-5
@test positive_sol_absval.errors[:l2] < 4.3e-5
@test positive_sol_absval.errors[:final] < 4.3e-18 # slightly better than general approach
@test general_sol_absval.t == positive_sol_absval.t

# specify abstol as array or scalar
positive_sol_absval2 = solve(prob_absval, BS3(); callback=PositiveDomain([1.0], abstol=[1e-6]))
@test positive_sol_absval.t == positive_sol_absval2.t &&
    positive_sol_absval.u == positive_sol_absval2.u
positive_sol_absval3 = solve(prob_absval, BS3(); callback=PositiveDomain([1.0], abstol=1e-6))
@test positive_sol_absval.t == positive_sol_absval3.t &&
    positive_sol_absval.u == positive_sol_absval3.u

# specify scalefactor
positive_sol_absval4 = solve(prob_absval, BS3(); callback=PositiveDomain([1.0], scalefactor=0.2))
@test length(positive_sol_absval.t) < length(positive_sol_absval4.t)
@test positive_sol_absval.errors[:l2] > positive_sol_absval4.errors[:l2]

"""
Knee problem

```math
\\frac{du}{dt} = \epsilon^{-1}(1-t-u)u
```

with initial condition ``u0=1``, and generally ``0 < \epsilon << 1``.
Here ``\epsilon=1e-6``. Then the solution approaches ``u=1-t`` for ``t<1``
and ``u=0`` for ``t>1``.
"""
function knee(t,u,du)
    du[1] = 1e6*(1-t-u[1])*u[1]
end

prob_knee = ODEProblem(knee, [1.0], (0.0, 2.0))

# unfortunately callbacks do not work with solver CVODE_BDF which is comparable to ode15s
# used in MATLAB example, so we use Rodas5
naive_sol_knee = solve(prob_knee, Rodas5())
@test naive_sol_knee[1, end] ≈ -1.0 atol=1e-5

# positive domain approach
positive_sol_knee = solve(prob_knee, Rodas5(); callback=PositiveDomain([1.0]))
@test all(x -> x[1] ≥ 0, positive_sol_knee.u)
@test positive_sol_knee[1, end] ≈ 0.0
