using DiffEqCallbacks
using Base.Test

# write your own tests here
tic()
@time @testset "AutoAbstol" begin include("autoabstol_tests.jl") end
@time @testset "Domain tests" begin include("domain_tests.jl") end
@time @testset "Manifold tests" begin include("manifold_tests.jl") end
@time @testset "StepsizeLimiter tests" begin include("stepsizelimiter_tests.jl") end
@time @testset "Saving tests" begin include("saving_tests.jl") end
@time @testset "Iterative tests" begin include("iterative_tests.jl") end
@time @testset "Periodic tests" begin include("periodic_tests.jl") end
toc()
