using OrdinaryDiffEq, Base.Test, DiffEqBase, DiffEqCallbacks

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

cb = ManifoldProjection(g)
sol = solve(prob,Vern7())
@test !(sol[end][1]^2 + sol[end][2]^2 ≈ 2)
sol = solve(prob,Vern7(),callback=cb)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2
