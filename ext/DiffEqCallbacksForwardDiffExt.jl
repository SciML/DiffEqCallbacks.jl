module DiffEqCallbacksForwardDiffExt

import DiffEqCallbacks: _autoabstol_value
using ForwardDiff: ForwardDiff

# Recursively unwrap `ForwardDiff.Dual` layers so `AutoAbstol` tracks the primal
# maximum state magnitude (tolerances stay at the plain float type).
_autoabstol_value(x::ForwardDiff.Dual) = _autoabstol_value(ForwardDiff.value(x))

end
