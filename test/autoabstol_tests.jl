using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqTsit5, Test, DiffEqBase, DiffEqCallbacks,
    SciMLBase
using ForwardDiff
import ODEProblemLibrary: prob_ode_2Dlinear, prob_ode_linear

prob = prob_ode_linear

cb = AutoAbstol()
integrator = init(prob, BS3(), callback = cb)
at1 = integrator.opts.abstol
step!(integrator)
at2 = integrator.opts.abstol
@test at1 < at2
step!(integrator)
at3 = integrator.opts.abstol
@test at2 < at3

solve(prob, BS3(), callback = cb)

prob2D = prob_ode_2Dlinear
cb2D = AutoAbstol(init_curmax = zeros(4, 2))
integrator2D_1 = init(prob2D, BS3(), callback = cb, abstol = 1.0e-6)
integrator2D_2 = init(prob2D, BS3(), callback = cb, abstol = fill(1.0e-6, 4, 2))
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)
step!(integrator2D_1)
step!(integrator2D_2)
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)
step!(integrator2D_1)
step!(integrator2D_2)
@test all(integrator2D_2.opts.abstol .== integrator2D_1.opts.abstol)

sol1 = solve(prob2D, BS3(), callback = cb2D, abstol = fill(1.0e-6, 4, 2))
sol2 = solve(
    prob2D, BS3(), callback = cb2D, abstol = fill(1.0e-6, 4, 2),
    reltol = fill(1.0e-3, 4, 2)
)
@test sol1.t == sol2.t && sol1.u == sol2.u
@test_throws MethodError solve(prob, BS3(), callback = cb2D, abstol = 1.0e-6)

# `AutoAbstol()` keeps a scalar `curmax` even for a vector state; its update must not allocate
function _osc!(du, u, p, t)
    du[1] = -0.001 * u[1] + u[2]
    du[2] = -u[1] - 0.001 * u[2]
    du[3] = -0.001 * u[3] + 2.0 * u[4]
    du[4] = -2.0 * u[3] - 0.001 * u[4]
    return nothing
end
function _autoabstol_affect_bytes()
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(_osc!, [1.0, 0.0, 0.5, 0.0], (0.0, 1.0e4))
    integ = init(prob, Tsit5(); callback = AutoAbstol())
    step!(integ)
    affect! = integ.opts.callback.discrete_callbacks[1].affect!
    affect!(integ)
    return @allocated affect!(integ)
end
@test _autoabstol_affect_bytes() == 0

# Differentiating through a solve that uses `AutoAbstol` must work: the tracked
# maximum stays at the primal float type instead of erroring on `Float64(::Dual)`
# (issue #351). The loss `sum(u(end))` with `du = -p*u` is `3*exp(-p)`, so the
# gradient at `p = 0.5` is `-3*exp(-0.5)`; the `1.0e-3` tolerance is loose
# relative to the solver's default tolerances.
function _autoabstol_forwarddiff_loss(p)
    f!(du, u, p, t) = (du .= -p[1] .* u; nothing)
    sol = solve(
        ODEProblem(f!, [1.0, 2.0], (0.0, 1.0), p), Tsit5(); callback = AutoAbstol()
    )
    return sum(sol.u[end])
end
@test ForwardDiff.gradient(_autoabstol_forwarddiff_loss, [0.5]) ≈ [-3 * exp(-0.5)] rtol = 1.0e-3

function _autoabstol_forwarddiff_array_loss(p)
    f!(du, u, p, t) = (du .= -p[1] .* u; nothing)
    sol = solve(
        ODEProblem(f!, [1.0, 2.0], (0.0, 1.0), p), Tsit5();
        callback = AutoAbstol(; init_curmax = zeros(2)), abstol = fill(1.0e-6, 2)
    )
    return sum(sol.u[end])
end
@test ForwardDiff.gradient(_autoabstol_forwarddiff_array_loss, [0.5]) ≈
    [-3 * exp(-0.5)] rtol = 1.0e-3

# Differentiating wrt the initial state `x` with an `init_curmax` derived from
# `x` must work: during AD `init_curmax` holds `Dual`s, which the constructor
# unwraps to the primal float type (issue #351). With `du = -0.5u`,
# `sum(u(end)) = (x[1] + x[2]) * exp(-0.5)`, so the gradient at `[1.0, 2.0]` is
# `[exp(-0.5), exp(-0.5)]`.
function _autoabstol_forwarddiff_u0_max_loss(x)
    f!(du, u, p, t) = (du .= -0.5 .* u; nothing)
    sol = solve(
        ODEProblem(f!, x, (0.0, 1.0)), Tsit5();
        callback = AutoAbstol(; init_curmax = maximum(abs, x))
    )
    return sum(sol.u[end])
end
@test ForwardDiff.gradient(_autoabstol_forwarddiff_u0_max_loss, [1.0, 2.0]) ≈
    fill(exp(-0.5), 2) rtol = 1.0e-3

function _autoabstol_forwarddiff_u0_abs_loss(x)
    f!(du, u, p, t) = (du .= -0.5 .* u; nothing)
    sol = solve(
        ODEProblem(f!, x, (0.0, 1.0)), Tsit5();
        callback = AutoAbstol(; init_curmax = abs.(x)),
        abstol = fill(1.0e-6, 2)
    )
    return sum(sol.u[end])
end
@test ForwardDiff.gradient(_autoabstol_forwarddiff_u0_abs_loss, [1.0, 2.0]) ≈
    fill(exp(-0.5), 2) rtol = 1.0e-3
