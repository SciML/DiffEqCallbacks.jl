# Timed Callbacks

The following callbacks are designed to help build callbacks with specific timing schemes
for when the `affect!` is to be detonated.

`PresetTimeCallback` accepts its `tstops` as a number, a vector of times, or a callable
`tstops(integrator)` returning a number or a vector of times. The callable is evaluated
when the callback is initialized at the start of each solve, so event times that depend
on the model parameters can be read from the integrator, for example
`PresetTimeCallback(integrator -> integrator.p, affect!)` when the parameters are the
event times. The same callback then works unchanged for a problem updated with
`remake(prob; p = ...)`, without rebuilding the callback for each parameter vector.

```@docs
PresetTimeCallback
PeriodicCallback
IterativeCallback
```
