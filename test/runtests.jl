using DiffEqCallbacks
import Functors
using Test
const GROUP = get(ENV, "GROUP", "All")

# write your own tests here
@time @testset "DiffEqCallbacks" begin
    if GROUP == "QA"
        @time @testset "Quality Assurance" begin
            include("qa.jl")
        end
    end

    if GROUP == "All" || GROUP == "Core"
        @time @testset "AutoAbstol" begin
            include("autoabstol_tests.jl")
        end
        @time @testset "TerminateSteadyState tests" begin
            include("terminatesteadystate_test.jl")
        end
        @time @testset "StepsizeLimiter tests" begin
            include("stepsizelimiter_tests.jl")
        end
        @time @testset "Function Calling tests" begin
            include("funccall_tests.jl")
        end
        @time @testset "IndependentlyLinearized tests" begin
            include("independentlylinearizedtests.jl")
        end
        @time @testset "PresetTime tests" begin
            include("preset_time.jl")
        end
        @time @testset "Iterative tests" begin
            include("iterative_tests.jl")
        end
        @time @testset "Periodic tests" begin
            include("periodic_tests.jl")
        end
        @time @testset "Manifold tests" begin
            include("manifold_tests.jl")
        end
        @time @testset "Domain tests" begin
            include("domain_tests.jl")
        end
        @time @testset "ProbInts tests" begin
            include("probints.jl")
        end
        @time @testset "Integrating tests" begin
            include("integrating_tests.jl")
        end
        @time @testset "Integrating sum tests" begin
            include("integrating_sum_tests.jl")
        end
        @time @testset "Integrating GK tests" begin
            include("integrating_GK_tests.jl")
        end
        @time @testset "Integrating GK Sum tests" begin
            include("integrating_GK_sum_tests.jl")
        end
        @time @testset "Saving tests" begin
            include("saving_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "NoPre" && isempty(VERSION.prerelease)
        import Pkg
        Pkg.activate("nopre")
        Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @testset "Integrating Sensitivity tests" begin
            include("nopre/integrating_sensitivity_tests.jl")
        end
        @time @testset "Integrating Sum Sensitivity tests" begin
            include("nopre/integrating_sum_sensitivity_tests.jl")
        end
        @time @testset "Saving Tracker tests" begin
            include("nopre/saving_tracker_tests.jl")
        end
    end
end
