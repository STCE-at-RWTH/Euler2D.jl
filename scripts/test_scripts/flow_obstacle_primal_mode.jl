using Revise

using BenchmarkTools
using Euler2D
using Euler2D: tangent_quadcell_list_and_id_grid, primal_quadcell_list_and_id_grid
using Euler2D: partition_cell_list, step_cell_simulation!
using OhMyThreads
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
bounds = ((-1.5, 0.0), (0.0, 2.0))
just_circle = [CircularObstacle((0.0, 0.0), 0.75)]
ncells = (540, 540 * 4 ÷ 3)

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
cell_partitions = partition_cell_list(global_cells, global_ids, 8; show_info = true);

using Euler2D: fast_partition_cell_list
fast_cell_partitions = fast_partition_cell_list(global_cells, global_ids, 4)

serial_update(partitions, bc, gas) =
    foreach(partitions) do p
        Euler2D.compute_partition_update_and_max_Δt!(p, bc, gas)
    end

parallel_update(partitions, bc, gas) =
    tforeach(partitions) do p
        Euler2D.compute_partition_update_and_max_Δt!(p, bc, gas)
    end

function serial_broadcast_update(partitions)
    needs_shared = Iterators.filter(
        ((a, b),) -> a ≠ b,
        ((a.id, b.id) for (a, b) in Iterators.product(partitions, partitions)),
    )
    foreach(needs_shared) do (id1, id2)
        Euler2D.propagate_updates_to!(partitions[id1], partitions[id2])
    end
end

function parallel_broadcast_update(partitions)
    needs_shared =
        Iterators.filter(
            tpl -> ≠(tpl...),
            ((a.id, b.id) for (a, b) in Iterators.product(partitions, partitions)),
        ) |> collect
    partition_locks = [Base.Lockable(p) for p ∈ partitions]
    tforeach(needs_shared) do (id1, id2)
        lock(partition_locks[id1])
        Euler2D.propagate_updates_to!(partition_locks[id1][], partitions[id2])
        unlock(partition_locks[id1])
    end
end

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
