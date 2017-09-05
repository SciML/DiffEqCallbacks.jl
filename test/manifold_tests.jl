using DiffEqCallbacks, OrdinaryDiffEq, RecursiveArrayTools, Base.Test

u0 = ones(2,2)
f = function (t,u,du)
  du[1,:] = u[2,:]
  du[2,:] = -u[1,:]
end
prob = ODEProblem(f,u0,(0.0,100.0))

function g(u,resid)
  resid[1] = u[2]^2 + u[1]^2 - 2
  resid[2] = u[3]^2 + u[4]^2 - 2
  resid[3] = 0
  resid[4] = 0
end

g_t(t,u,resid) = g(u,resid)

isautonomous(p::ManifoldProjection{autonomous,NL}) where {autonomous,NL} = autonomous

sol = solve(prob,Vern7())
@test !(sol[end][1]^2 + sol[end][2]^2 ≈ 2)

cb = ManifoldProjection(g)
@test isautonomous(cb.affect!)
solve(prob,Vern7(),callback=cb)
@time sol=solve(prob,Vern7(),callback=cb)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2

cb_t = ManifoldProjection(g_t)
@test !isautonomous(cb_t.affect!)
solve(prob,Vern7(),callback=cb_t)
@time sol_t = solve(prob,Vern7(),callback=cb_t)
@test sol_t.u == sol.u && sol_t.t == sol.t

# test array partitions

u₀ = ArrayPartition(ones(2), ones(2))
prob = ODEProblem(f, u₀, (0.0, 100.0))

sol = solve(prob,Vern7(),callback=cb)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2
