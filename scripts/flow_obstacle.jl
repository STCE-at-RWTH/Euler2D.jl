using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

bcs = (ntuple(Returns(ExtrapolateToPhantom()), 4)..., StrongWall())
bounds = ((0.0, 1.0), (0.0, 1.0))
ncells = (500,500)
(xs, ys) = map(zip(bounds, ncells)) do (b, n)
    v = range(b...; length = n + 1)
    return v[1:end-1] .+ step(v) / 2
end

active_cells, active_list = quadcell_list_and_id_grid(bounds, ncells, []) do (x, y)
    ConservedProps(1.0, (0.0, 1.0), 1.0e5)
end

active_cells_next = similar(active_cells)
