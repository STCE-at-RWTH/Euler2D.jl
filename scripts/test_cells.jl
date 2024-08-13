using BenchmarkTools
using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

using Euler2D: active_cell_mask, active_cell_ids_from_mask, cell_neighbor_status
using Euler2D: quadcell_list_and_id_grid, phantom_neighbor, single_cell_neighbor_data, split_axis, partition_cell_list, expand_to_neighbors
using Euler2D: compute_cell_update, compute_partition_update, compute_next_u
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
ncells = (100, 100)
active_cells, active_ids = quadcell_list_and_id_grid(bounds, ncells, obstacle) do (x, y)
    x <= 0 ? ambient : amb2
end;

test_partition = partition_cell_list(active_cells, active_ids, 4);
##

ndata = single_cell_neighbor_data(2, active_cells, bcs, DRY_AIR)
@benchmark single_cell_neighbor_data(1, $active_cells, $bcs, $DRY_AIR)

pndata = phantom_neighbor(1, active_cells, :south, ExtrapolateToPhantom(), DRY_AIR)
@benchmark phantom_neighbor(1, $active_cells, :north, ExtrapolateToPhantom(), DRY_AIR)

# @benchmark compute_cell_update($cdata, $ndata, 0.01, 0.01, DRY_AIR)

ndata2 = map(state_to_vector, ndata)
compute_cell_update(state_to_vector(active_cells[2].u), ndata2, 0.01, 0.01, DRY_AIR)
@benchmark compute_cell_update($(state_to_vector(active_cells[2].u)), $ndata2, 0.01, 0.01, DRY_AIR)