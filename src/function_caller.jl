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
            if typeof(integrator.u) <: Union{Number, StaticArraysCore.SArray}
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

function functioncalling_initialize(cb, affect!, u, t, integrator)    
    if affect!.funciter != 0
        if integrator.tdir > 0
            affect!.funcat = BinaryMinHeap(affect!.funcat_cache)
        else
            affect!.funcat = BinaryMaxHeap(affect!.funcat_cache)
        end
        affect!.funciter = 0
    end
    affect!.func_start && affect!(integrator)
end

"""
```julia
FunctionCallingCallback(func;
               funcat=Vector{Float64}(),
               func_everystep=isempty(funcat),
               func_start = true
               tdir=1)
```

The function calling callback lets you define a function `func(u,t,integrator)`
which gets calls at the time points of interest. The constructor is:

- `func(t, u, integrator)` is the function to be called.
- `funcat` values that the function is sure to be evaluated at.
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
    funcat_vec = collect(funcat)
    if tdir > 0
        funcat_internal = BinaryMinHeap(funcat_vec)
    else
        funcat_internal = BinaryMaxHeap(funcat_vec)
    end
    affect! = FunctionCallingAffect(func, funcat_internal,
                                    funcat_vec, func_everystep, func_start, 0)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(condtion, affect!;
                     initialize = functioncalling_initialize,
                     save_positions = (false, false))
end

export FunctionCallingCallback
