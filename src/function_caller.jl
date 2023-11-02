mutable struct FunctionCallingAffect{funcFunc, funcatType, funcatCacheType}
    func::funcFunc
    funcat::funcatType
    funcat_cache::funcatCacheType
    func_everystep::Bool
    func_start::Bool
    funciter::Int
end

function (affect!::FunctionCallingAffect)(integrator, force_func = false)
    # see OrdinaryDiffEq.jl -> integrator_utils.jl, function funcvalues!
    while !isempty(affect!.funcat) &&
        integrator.tdir * first(affect!.funcat) <= integrator.tdir * integrator.t # Perform funcat
        affect!.funciter += 1
        curt = pop!(affect!.funcat) # current time
        if curt != integrator.t # If <t, interpolate
            if integrator isa SciMLBase.AbstractODEIntegrator
                # Expand lazy dense for interpolation
                DiffEqBase.addsteps!(integrator)
            end
            if integrator.u isa Union{Number, StaticArraysCore.SArray}
                curu = integrator(curt)
            else
                curu = first(get_tmp_cache(integrator))
                integrator(curu, curt) # inplace since func_func allocates
            end
            affect!.func(curu, curt, integrator)
        else # ==t, just func
            affect!.func(integrator.u, integrator.t, integrator)
        end
    end
    if affect!.func_everystep || force_func
        affect!.funciter += 1
        affect!.func(integrator.u, integrator.t, integrator)
    end
    u_modified!(integrator, false)
end

function functioncalling_initialize(cb, u, t, integrator)
    funcat_cache = cb.affect!.funcat_cache
    if cb.affect!.funciter != 0 || funcat_cache isa Number
        tspan = integrator.sol.prob.tspan
        funcat_cache = cb.affect!.funcat_cache
        funcat_vec = if funcat_cache isa Number
            step = funcat_cache
            t0, tf = tspan
            if !cb.affect!.func_start
                t0 += step
            end
            range(t0, tf; step)
        else
            funcat_cache
        end
        if integrator.tdir > 0
            cb.affect!.funcat = BinaryMinHeap(funcat_vec)
        else
            cb.affect!.funcat = BinaryMaxHeap(funcat_vec)
        end
        cb.affect!.funciter = 0
    end
    cb.affect!.func_start && cb.affect!(integrator)
    u_modified!(integrator, false)
end

"""
```julia
FunctionCallingCallback(func;
    funcat = Vector{Float64}(),
    func_everystep = isempty(funcat),
    func_start = true,
    tdir = 1)
```

The function calling callback lets you define a function `func(u,t,integrator)`
which gets calls at the time points of interest. The constructor is:

  - `func(u, t, integrator)` is the function to be called.
  - `funcat` values or interval that the function is sure to be evaluated at.
  - `func_everystep` whether to call the function after each integrator step.
  - `func_start` whether the function is called the initial condition.
  - `tdir` should be `sign(tspan[end]-tspan[1])`. It defaults to `1` and should
    be adapted if `tspan[1] > tspan[end]`.
"""
function FunctionCallingCallback(func;
    funcat = Vector{Float64}(),
    func_everystep = isempty(funcat),
    func_start = true,
    tdir = 1)
    # funcat conversions, see OrdinaryDiffEq.jl -> integrators/type.jl
    if funcat isa Number
        # expand to range using tspan in functioncalling_initialize
        funcat_cache = funcat
        funcat_heap = fill(funcat, 0)
    else
        funcat_heap = funcat_cache = collect(funcat)
    end

    if tdir > 0
        funcat_internal = BinaryMinHeap(funcat_heap)
    else
        funcat_internal = BinaryMaxHeap(funcat_heap)
    end
    affect! = FunctionCallingAffect(func, funcat_internal,
        funcat_cache, func_everystep, func_start, 0)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(condtion, affect!;
        initialize = functioncalling_initialize,
        save_positions = (false, false))
end

export FunctionCallingCallback
