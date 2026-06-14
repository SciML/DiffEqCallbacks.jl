using SciMLTesting

# The NoPre group is selectable by name only on non-prerelease Julia (the original
# `GROUP == "NoPre" && isempty(VERSION.prerelease)` gate). Under "All" the NoPre
# folder always ran regardless of prerelease, and folder-discovery's "All" globs the
# NoPre folder unconditionally, so "All" is unaffected by this guard.
if current_group() == "NoPre" && !isempty(VERSION.prerelease)
    # prerelease Julia: GROUP=NoPre runs nothing, matching the original gate.
else
    run_tests()
end
