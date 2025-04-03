using OrdinaryDiffEqVerner, Test, DiffEqBase, DiffEqCallbacks, RecursiveArrayTools, NonlinearSolve
using ForwardDiff, ADTypes

u0 = ones(2, 2)
f = function (du, u, p, t)
    du[1, :] = u[2, :]
    du[2, :] = -u[1, :]
end
prob = ODEProblem(f, u0, (0.0, 100.0))

function g(resid, u, p)
    resid[1] = u[2]^2 + u[1]^2 - 2
    resid[2] = u[3]^2 + u[4]^2 - 2
end

g_t(resid, u, p, t) = g(resid, u, p)

sol = solve(prob, Vern7())
@test !(sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2)

# autodiff=true
@inferred ManifoldProjection(g; autonomous = Val(false), resid_prototype = zeros(2))
cb = ManifoldProjection(g; resid_prototype = zeros(2), autodiff = AutoForwardDiff())
solve(prob, Vern7(), callback = cb)
@time sol = solve(prob, Vern7(), callback = cb)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

cb_t = ManifoldProjection(g_t; resid_prototype = zeros(2), autodiff = AutoForwardDiff())
solve(prob, Vern7(), callback = cb_t)
@time sol_t = solve(prob, Vern7(), callback = cb_t)
@test sol_t.u == sol.u && sol_t.t == sol.t

# autodiff=false
cb_false = ManifoldProjection(
    g; nlsolve = NewtonRaphson(; autodiff = AutoFiniteDiff()), resid_prototype = zeros(2),
    autodiff = AutoFiniteDiff())
solve(prob, Vern7(), callback = cb_false)
sol = solve(prob, Vern7(), callback = cb_false)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

cb_t_false = ManifoldProjection(g_t,
    nlsolve = NewtonRaphson(; autodiff = AutoFiniteDiff()), resid_prototype = zeros(2),
    autodiff = AutoFiniteDiff())
solve(prob, Vern7(), callback = cb_t_false)
sol_t = solve(prob, Vern7(), callback = cb_t_false)
@test sol_t.u == sol.u && sol_t.t == sol.t

# test array partitions
function f_ap!(du, u, p, t)
    du[1:2] .= u[3:4]
    du[3:4] .= u[1:2]
end

u₀ = ArrayPartition(ones(2), ones(2))
prob = ODEProblem(f_ap!, u₀, (0.0, 100.0))

sol = solve(prob, Vern7(), callback = cb)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

sol = solve(prob, Vern7(), callback = cb_t)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

sol = solve(prob, Vern7(), callback = cb_false)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

sol = solve(prob, Vern7(), callback = cb_t_false)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

# Test termination if cannot project to manifold
function g_unsat(resid, u, p)
    resid[1] = u[2]^2 + u[1]^2 - 1000
    resid[2] = u[2]^2 + u[1]^2 - 20
end

cb_unsat = ManifoldProjection(
    g_unsat; resid_prototype = zeros(2), autodiff = AutoForwardDiff())
sol = solve(prob, Vern7(), callback = cb_unsat)
@test !SciMLBase.successful_retcode(sol)
@test last(sol.t) != 100.0

cb_unsat = ManifoldProjection(
    g_unsat; resid_prototype = zeros(2), autodiff = AutoForwardDiff(), nlsolve = NewtonRaphson())
sol = solve(prob, Vern7(), callback = cb_unsat)
@test !SciMLBase.successful_retcode(sol)
@test last(sol.t) != 100.0

# Tests for OOP Manifold Projection
function g_oop(u, p)
    return [u[2]^2 + u[1]^2 - 2
            u[3]^2 + u[4]^2 - 2]
end

g_oop_t(u, p, t) = g_oop(u, p)

f_oop = function (u, p, t)
    return stack((u[2, :], -u[1, :]))
end
prob = ODEProblem(f_oop, u0, (0.0, 100.0))

# autodiff=true
@inferred ManifoldProjection(g_oop; autonomous = Val(false))
cb = ManifoldProjection(g_oop; autodiff = AutoForwardDiff())
solve(prob, Vern7(), callback = cb)
@time sol = solve(prob, Vern7(), callback = cb)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

cb_t = ManifoldProjection(g_oop_t; autodiff = AutoForwardDiff())
solve(prob, Vern7(), callback = cb_t)
@time sol_t = solve(prob, Vern7(), callback = cb_t)
@test sol_t.u == sol.u && sol_t.t == sol.t

# autodiff=false
cb_false = ManifoldProjection(
    g_oop; nlsolve = NewtonRaphson(; autodiff = AutoFiniteDiff()), autodiff = AutoFiniteDiff())
solve(prob, Vern7(), callback = cb_false)
sol = solve(prob, Vern7(), callback = cb_false)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

cb_t_false = ManifoldProjection(g_oop_t,
    nlsolve = NewtonRaphson(; autodiff = AutoFiniteDiff()), autodiff = AutoFiniteDiff())
solve(prob, Vern7(), callback = cb_t_false)
sol_t = solve(prob, Vern7(), callback = cb_t_false)
@test sol_t.u == sol.u && sol_t.t == sol.t

# test array partitions
f_ap(u, p, t) = ArrayPartition(u[3:4], u[1:2])

u₀ = ArrayPartition(ones(2), ones(2))
prob = ODEProblem(f_ap, u₀, (0.0, 100.0))

sol = solve(prob, Vern7(), callback = cb)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

sol = solve(prob, Vern7(), callback = cb_t)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

sol = solve(prob, Vern7(), callback = cb_false)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2

sol = solve(prob, Vern7(), callback = cb_t_false)
@test sol.u[end][1]^2 + sol.u[end][2]^2 ≈ 2
