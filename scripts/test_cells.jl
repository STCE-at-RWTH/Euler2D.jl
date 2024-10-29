using BenchmarkTools
using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

using Euler2D: active_cell_mask, active_cell_ids_from_mask, cell_neighbor_status
using Euler2D: quadcell_list_and_id_grid, phantom_neighbor, neighbor_cells, split_axis, partition_cell_list, expand_to_neighbors
using Euler2D: compute_cell_update_and_max_Δt, compute_partition_update_and_max_Δt!, apply_partition_update!
using Euler2D: step_cell_simulation!

function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ambient = launder_units(PrimitiveProps(0.662, (4.0, 0.0), 220.0))
amb2 = launder_units(PrimitiveProps(0.662, (-4.0, 0.0), 220.0))

bc_right = SupersonicInflow(ambient, DRY_AIR)
bcs = (
    ExtrapolateToPhantom(), # north
    ExtrapolateToPhantom(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(),
)
bounds = ((-4.0, 4.0), (-4.0, 4.0))
obstacle = [CircularObstacle((0.0, 0.0), 0.75)]
ncells = (800, 800)
active_cells, active_ids = quadcell_list_and_id_grid(bounds, ncells, obstacle) do (x, y)
    x <= 0 ? ambient : amb2
end;

test_partition = partition_cell_list(active_cells, active_ids, 4);
##

cell1 = active_cells[1]
@benchmark phantom_neighbor($cell1, :north, $(bcs[1]), DRY_AIR)
@benchmark neighbor_cells($cell1, $active_cells, $bcs, $DRY_AIR)

##

@code_warntype compute_cell_update_and_max_Δt(cell1, active_cells, bcs, DRY_AIR)
nbrs = neighbor_cells(cell1, active_cells, bcs, DRY_AIR)
f(cell, neighbors) = begin
    ifaces = (
        north = (2, cell, neighbors.north),
        south = (2, neighbors.south, cell),
        east = (1, cell, neighbors.east),
        west = (1, neighbors.west, cell),
    )
    return ifaces
end
ifaces1 = f(cell1, nbrs)
@benchmark Euler2D._iface_speed($(ifaces1[1]), $DRY_AIR)
@benchmark Euler2D.maximum_cell_signal_speeds($ifaces1, $DRY_AIR)
@benchmark compute_cell_update_and_max_Δt($cell1, $active_cells, $bcs, $DRY_AIR)
@benchmark compute_partition_update_and_max_Δt!($(test_partition[1]), $bcs, $DRY_AIR)

@code_warntype step_cell_simulation!(test_partition, 0.1, bcs, 0.8, DRY_AIR)
@benchmark step_cell_simulation!($test_partition, 0.1, $bcs, 0.8, $DRY_AIR)