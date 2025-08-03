# Local test problem definitions to avoid circular dependency with ODEProblemLibrary
# These replace the imports from ODEProblemLibrary to break the circular dependency:
# ModelingToolkit → DiffEqCallbacks → ODEProblemLibrary → ModelingToolkit

using SciMLBase

# Linear ODE: du/dt = u, u(0) = 1/2
function linear_f(u, p, t)
    return u
end
function linear_f!(du, u, p, t)
    du[1] = u[1]
end

prob_ode_linear = ODEProblem(linear_f, 1/2, (0.0, 1.0))

# 2D Linear ODE: du/dt = A*u, u(0) = [1.0, 1.0]
function linear_2d_f!(du, u, p, t)
    du[1] = u[1] + u[2]
    du[2] = u[1] - u[2]  
end
function linear_2d_f(u, p, t)
    return [u[1] + u[2], u[1] - u[2]]
end

prob_ode_2Dlinear = ODEProblem(linear_2d_f!, [1.0, 1.0], (0.0, 1.0))

# Van der Pol oscillator: d²x/dt² - μ(1-x²)dx/dt + x = 0
function vanderpol_f!(du, u, p, t)
    μ = 1.0
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
end

prob_ode_vanderpol = ODEProblem(vanderpol_f!, [2.0, 0.0], (0.0, 6.3))

# Rigid body dynamics
function rigidbody_f!(du, u, p, t)
    du[1] = u[2] * u[3]
    du[2] = -u[1] * u[3]
    du[3] = -0.51 * u[1] * u[2]
end

prob_ode_rigidbody = ODEProblem(rigidbody_f!, [1.0, 0.0, 0.9], (0.0, 20.0))

# Nonlinear chemistry (Belousov-Zhabotinsky reaction simplified)
function nonlinchem_f!(du, u, p, t)
    du[1] = -0.04 * u[1] + 1e4 * u[2] * u[3]
    du[2] = 0.04 * u[1] - 1e4 * u[2] * u[3] - 3e7 * u[2]^2
    du[3] = 3e7 * u[2]^2
end

prob_ode_nonlinchem = ODEProblem(nonlinchem_f!, [1.0, 0.0, 0.0], (0.0, 4e5))

# Lorenz system
function lorenz_f!(du, u, p, t)
    σ, ρ, β = 10.0, 28.0, 8/3
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]  
    du[3] = u[1] * u[2] - β * u[3]
end

prob_ode_lorenz = ODEProblem(lorenz_f!, [1.0, 0.0, 0.0], (0.0, 100.0))