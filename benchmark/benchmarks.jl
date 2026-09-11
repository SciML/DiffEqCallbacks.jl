using DiffEqCallbacks, OrdinaryDiffEq, BenchmarkTools

const SUITE = BenchmarkGroup()

# Simple decay problem
f_decay!(du, u, p, t) = (du .= -u)
prob = ODEProblem(f_decay!, ones(5), (0.0, 10.0))

# =============================================================================
# Event callbacks
# =============================================================================

SUITE["callbacks"] = BenchmarkGroup()

# ContinuousCallback: root-finding when u[1] crosses 0.5
condition(u, t, integrator) = u[1] - 0.5
function affect!(integrator)
    return integrator.u[1] += 0.1
end
cb_cont = ContinuousCallback(condition, affect!)
SUITE["callbacks"]["continuous"] = @benchmarkable solve(
    $prob, Tsit5(); callback = $cb_cont
)

# PresetTimeCallback
function preset_affect!(integrator)
    return integrator.u .+= 0.1
end
cb_preset = PresetTimeCallback([2.0, 5.0, 8.0], preset_affect!)
SUITE["callbacks"]["preset_time"] = @benchmarkable solve(
    $prob, Tsit5(); callback = $cb_preset
)

# TerminateSteadyState
cb_term = TerminateSteadyState(1.0e-8)
SUITE["callbacks"]["terminate_steady"] = @benchmarkable solve(
    $prob, Tsit5(); callback = $cb_term
)

# SavingCallback with SavedValues
saved = SavedValues(Float64, Vector{Float64})
save_fn(u, t, integrator) = copy(u)
cb_save = SavingCallback(save_fn, saved; saveat = 0:0.1:10)
SUITE["callbacks"]["saving"] = @benchmarkable solve(
    $prob, Tsit5(); callback = $cb_save
)

# PeriodicCallback
cb_periodic = PeriodicCallback(preset_affect!, 1.0)
SUITE["callbacks"]["periodic"] = @benchmarkable solve(
    $prob, Tsit5(); callback = $cb_periodic
)
