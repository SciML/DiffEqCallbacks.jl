using Pkg
println("Testing DiffEqBase version compatibility...")

try
    # Update to latest DiffEqBase  
    Pkg.add(name = "DiffEqBase", version = "6.180")
    println("Successfully updated to DiffEqBase v6.180")

    # Test adding DataStructures v0.19
    Pkg.add(name = "DataStructures", version = "0.19")
    println("Successfully added DataStructures v0.19")

    # Test that everything works together
    using DiffEqBase, DataStructures, DiffEqCallbacks
    println("All packages load successfully together")

catch e
    println("Error: ", e)
end
