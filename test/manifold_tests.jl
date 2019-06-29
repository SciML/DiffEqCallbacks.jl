using OrdinaryDiffEq, Test, DiffEqBase, DiffEqCallbacks, RecursiveArrayTools, ManifoldProjections

u0 = ones(2,2)
f = function (du,u,p,t)
  du[1,:] = u[2,:]
  du[2,:] = -u[1,:]
end
prob = ODEProblem(f,u0,(0.0,100.0))

function g(resid,u)
  resid[1] = u[2]^2 + u[1]^2 - 2
  resid[2] = u[3]^2 + u[4]^2 - 2
  resid[3] = 0
  resid[4] = 0
end

g_t(resid,u,p,t) = g(resid,u)

isautonomous(p::ManifoldProjection{autonomous,NL}) where {autonomous,NL} = autonomous

sol = solve(prob,Vern7())
@test !(sol[end][1]^2 + sol[end][2]^2 ≈ 2)

# autodiff=true
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

# autodiff=false
cb_false = ManifoldProjection(g, nlsolve=OrdinaryDiffEq.NLSOLVEJL_SETUP(autodiff=false))
@test isautonomous(cb_false.affect!)
solve(prob,Vern7(),callback=cb_false)
sol=solve(prob,Vern7(),callback=cb_false)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2

cb_t_false = ManifoldProjection(g_t, nlsolve=OrdinaryDiffEq.NLSOLVEJL_SETUP(autodiff=false))
@test !isautonomous(cb_t_false.affect!)
solve(prob,Vern7(),callback=cb_t_false)
sol_t = solve(prob,Vern7(),callback=cb_t_false)
@test sol_t.u == sol.u && sol_t.t == sol.t

# test array partitions
u₀ = ArrayPartition(ones(2), ones(2))
prob = ODEProblem(f, u₀, (0.0, 100.0))

sol = solve(prob,Vern7(),callback=cb)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2

sol = solve(prob,Vern7(),callback=cb_t)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2

# does not work since Calculus.jl (on which NLsolve.jl depends)
# implements only Jacobians of vectors
sol = solve(prob,Vern7(),callback=cb_false)
sol[end][1]^2 + sol[end][2]^2 ≈ 2

sol = solve(prob,Vern7(),callback=cb_t_false)
sol[end][1]^2 + sol[end][2]^2 ≈ 2

# Test using ManifoldProjections.jl
# Test the equations above now transformed to the complex plane
u0 = (1+1im) * ones(ComplexF64, 2)
function f(du,u,p,t)
  @. du[:] = im * u
end
prob = ODEProblem(f,u0,(0.0,100.0))

# Each of the
S = Sphere(sqrt(2))
M = PowerManifold(S, (1,), (2,))

sol = solve(prob,Vern7())
@test !(abs2(sol[end][1]) ≈ 2)

cb = ManifoldProjection(M)
# @test isautonomous(cb.affect!)
solve(prob,Vern7(),callback=cb)
@time sol=solve(prob,Vern7(),callback=cb)
@test all(abs2.(sol[end]) .≈ 2)

# test array partitions
u₀ = ArrayPartition([1.0+im], [1.0+im])
prob = ODEProblem(f, u₀, (0.0, 100.0))

sol = solve(prob,Vern7(),callback=cb)
@test all(abs2.(sol[end]) .≈ 2)
