using BenchmarkTools
using Euler2D
using Euler2D: tangent_quadcell_list_and_id_grid, primal_quadcell_list_and_id_grid
using Euler2D: partition_cell_list, step_cell_simulation!
using ForwardDiff
using LinearAlgebra
using PProf
using Profile
using ShockwaveProperties
using StaticArrays
using Unitful

bcs = (
    ExtrapolateToPhantom(), # north 
    ExtrapolateToPhantom(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)
bounds = ((-2.0, 0.0), (-1.5, 1.5))
just_circle = [CircularObstacle((0.0, 0.0), 0.75)]
ncells = (50, 75)

starting_parameters = SVector(0.662, 4.0, 220.0)

function u0(x, p)
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

ambient = u0(nothing, starting_parameters)

x0 = 1.0u"m"
a0 = speed_of_sound(ambient, DRY_AIR)
ρ0 = density(ambient)
scale = EulerEqnsScaling(x0, ρ0, a0)

#
# # compile and run everything.
#
global_cells, global_ids = primal_quadcell_list_and_id_grid(
    u0,
    starting_parameters,
    bounds,
    ncells,
    scale,
    just_circle,
);
cell_partitions = partition_cell_list(global_cells, global_ids, 2; show_info = true);

using Euler2D: fast_partition_cell_list
# fast_cell_partitions = fast_partition_cell_list(global_cells, global_ids, 2)
# step_cell_simulation!(cell_partitions, 0.1, bcs, 0.5, DRY_AIR)
#
# ## 
# # now profile it
# cell_partitions = partition_cell_list(global_cells, global_ids, 2);
# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate = 0.5 begin
#     step_cell_simulation!(cell_partitions, 0.1, bcs, 0.5, DRY_AIR)
# end
# prof = Profile.Allocs.fetch();
#
# ##
#
# PProf.Allocs.pprof(prof; from_c = false)
#
# # check with Benchmarktools?
# ##
# cell_partitions2 = partition_cell_list(global_cells, global_ids, 2);
# @benchmark step_cell_simulation!($cell_partitions2, 0.1, $bcs, 0.5, $DRY_AIR)
#
# # note: 18.11.24 -- killed allocations on primal mode.
