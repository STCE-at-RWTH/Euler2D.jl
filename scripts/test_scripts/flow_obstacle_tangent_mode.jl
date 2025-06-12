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

starting_parameters = SVector(0.662, 4.0, 220.0, 0.75)

function u0(x, p)
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

ambient = u0(nothing, starting_parameters)

x0 = 1.0u"m"
a0 = speed_of_sound(ambient, DRY_AIR)
ρ0 = density(ambient)
scale = EulerEqnsScaling(x0, ρ0, a0)


# compile and run everything.

global_cells, global_ids = tangent_quadcell_list_and_id_grid(u0, starting_parameters, bounds, ncells, scale, just_circle);
@assert isconcretetype(valtype(global_cells))
cell_partitions = partition_cell_list(global_cells, global_ids, 2);
step_cell_simulation!(cell_partitions, 0.1, bcs, 0.5, DRY_AIR)

## 
# now profile it
cell_partitions = partition_cell_list(global_cells, global_ids, 2);
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1 begin
    step_cell_simulation!(cell_partitions, 0.1, bcs, 0.5, DRY_AIR)
end
prof = Profile.Allocs.fetch();

##

PProf.Allocs.pprof(prof; from_c = false)

# check with Benchmarktools?
##
using Euler2D: neighbor_cells, phantom_neighbor, flip_velocity
using Euler2D: compute_cell_update_and_max_Δt, maximum_cell_signal_speeds
using Euler2D: ϕ_hll, ϕ_hll_jvp

cell_partitions2 = partition_cell_list(global_cells, global_ids, 2);
@benchmark step_cell_simulation!($cell_partitions2, 0.1, $bcs, 0.5, $DRY_AIR)

part1 = cell_partitions2[1];
cell1 = part1.cells_map[1]
neighbors1 = neighbor_cells(cell1, (part1.cells_map), bcs, DRY_AIR)
ifaces1 = (
    north = (2, cell1, neighbors1.north),
    south = (2, neighbors1.south, cell1),
    east = (1, cell1, neighbors1.east),
    west = (1, neighbors1.west, cell1),
)

maxdt(cell, ifaces, gas) = begin
    a = maximum_cell_signal_speeds(ifaces, gas)
    Δt_max = min((cell.extent ./ a)...)
    return Δt_max
end

@benchmark maxdt($cell1, $ifaces1, $DRY_AIR)

flux(ifaces, gas) = map(ifaces) do (dim, cell_L, cell_R)
    return (
        ϕ_hll(cell_L.u, cell_R.u, dim, gas),
        ϕ_hll_jvp(cell_L.u, cell_L.u̇, cell_R.u, cell_R.u̇, dim, gas),
    )
end

@benchmark flux($ifaces1, $DRY_AIR)

@benchmark neighbor_cells($cell1, $(part1.cells_map), $bcs, $DRY_AIR)
@code_warntype neighbor_cells(cell1, (part1.cells_map), bcs, DRY_AIR)

@code_warntype compute_cell_update_and_max_Δt(cell1, (part1.cells_map), bcs, DRY_AIR)
@benchmark compute_cell_update_and_max_Δt($cell1, $(part1.cells_map), $bcs, $DRY_AIR)

@code_warntype phantom_neighbor(cell1, :south, bcs[2], DRY_AIR)
@code_warntype flip_velocity(cell1.u̇, 2)

@benchmark phantom_neighbor($cell1, :south, $(bcs[2]), $DRY_AIR)