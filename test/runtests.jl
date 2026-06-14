using DiffEqCallbacks
import Functors
using Test
using SafeTestsets
const GROUP = get(ENV, "GROUP", "All")

# write your own tests here
@time @testset "DiffEqCallbacks" begin
    if GROUP == "QA"
        @time @safetestset "Quality Assurance" begin
            include("qa.jl")
        end
    end

    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "AutoAbstol" begin
            include("autoabstol_tests.jl")
        end
        @time @safetestset "TerminateSteadyState tests" begin
            include("terminatesteadystate_test.jl")
        end
        @time @safetestset "StepsizeLimiter tests" begin
            include("stepsizelimiter_tests.jl")
        end
        @time @safetestset "Function Calling tests" begin
            include("funccall_tests.jl")
        end
        @time @safetestset "IndependentlyLinearized tests" begin
            include("independentlylinearizedtests.jl")
        end
        @time @safetestset "PresetTime tests" begin
            include("preset_time.jl")
        end
        @time @safetestset "Iterative tests" begin
            include("iterative_tests.jl")
        end
        @time @safetestset "Periodic tests" begin
            include("periodic_tests.jl")
        end
        @time @safetestset "Manifold tests" begin
            include("manifold_tests.jl")
        end
        @time @safetestset "Domain tests" begin
            include("domain_tests.jl")
        end
        @time @safetestset "ProbInts tests" begin
            include("probints.jl")
        end
        @time @safetestset "Integrating tests" begin
            include("integrating_tests.jl")
        end
        @time @safetestset "Integrating sum tests" begin
            include("integrating_sum_tests.jl")
        end
        @time @safetestset "Integrating GK tests" begin
            include("integrating_GK_tests.jl")
        end
        @time @safetestset "Integrating GK Sum tests" begin
            include("integrating_GK_sum_tests.jl")
        end
        @time @safetestset "Saving tests" begin
            include("saving_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "NoPre" && isempty(VERSION.prerelease)
        import Pkg
        Pkg.activate("nopre")
        Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "JET tests" begin
            include("nopre/jet_tests.jl")
        end
        @time @safetestset "Integrating Sensitivity tests" begin
            include("nopre/integrating_sensitivity_tests.jl")
        end
        @time @safetestset "Integrating Sum Sensitivity tests" begin
            include("nopre/integrating_sum_sensitivity_tests.jl")
        end
        @time @safetestset "Saving Tracker tests" begin
            include("nopre/saving_tracker_tests.jl")
        end
    end
end
