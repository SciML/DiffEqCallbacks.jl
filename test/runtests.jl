using SciMLTesting

# The AD group is selectable by name only on non-prerelease Julia (the AD/sensitivity
# stack — SciMLSensitivity, Tracker, Zygote — does not support prerelease). Under "All"
# the AD folder always ran regardless of prerelease, and folder-discovery's "All" globs
# the AD folder unconditionally, so "All" is unaffected by this guard.
if current_group() == "AD" && !isempty(VERSION.prerelease)
    # prerelease Julia: GROUP=AD runs nothing, matching the original NoPre gate.
else
    run_tests()
end
