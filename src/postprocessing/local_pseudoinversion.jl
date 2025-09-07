module LocalPseudoinversion

using Accessors
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using LinearAlgebra
using StaticArrays

using Euler2D
using Euler2D: CellNeighboring, FVMCell, TangentQuadCell
using Euler2D: fdiff_eps
# TODO organize exports from Euler2D
using Euler2D: minimum_cell_size, all_cells_overlapping, total_mass_contained_by
using Euler2D: ∇u_at
using PlanePolygons

const fdiff_backend = AutoForwardDiff()

### SOME POLYGON STUFF
# To my future self: sorry

function intersection_point_jacobian(point, poly1, poly2)
    N = num_vertices(poly1)
    J = zeros(eltype(point), (2, 2 * N))
    foreach(enumerate(zip(edge_starts(poly1), edge_ends(poly1)))) do (i, (p1, p2))
        if is_other_point_on_line(Line(p1, p2 - p1), point)
            for ell2 ∈
                Iterators.filter(ℓ -> is_other_point_on_line(ℓ, point), edge_lines(poly2))
                jac = DifferentiationInterface.jacobian(fdiff_backend, vcat(p1, p2)) do vals
                    q1 = vals[SVector(1, 2)]
                    q2 = vals[SVector(3, 4)]
                    return line_intersect(Line(q1, q2 - q1), ell2)
                end
                j = ((i + 1) % N) + 1
                J[2*i-1:2*i] = jac[SVector(1, 2)]
                J[2*j-1:2*j] = jac[SVector(3, 4)]
            end
        end
    end
    return J
end

# gets the jacobian of the intersection area w.r.t. the first argument
# allocates like hell, but... it works and isn't performance critical?.
function intersection_area_jacobian(flat_poly1, poly2)
    grad1 = zero(flat_poly1)
    for i in eachindex(flat_poly1)
        h = fdiff_eps(flat_poly1[i])
        in1 = @set flat_poly1[i] += h
        in2 = @set flat_poly1[i] -= h
        out1 = poly_area(poly_intersection(in1, poly2))
        out2 = poly_area(poly_intersection(in2, poly2))
        @reset grad1[i] = (out1 - out2) / (2 * h)
    end
    return grad1
end

### END POLYGON STUFF

"""
    CoarsePolyCell{T, NV}
"""
struct CoarsePolyCell{T,NV,NSEEDS,NTANGENTS} <: FVMCell{T}
    id::Int
    boundary::SClosedPolygon{T,NV}
    u::SVector{4,T}
    u̇::SMatrix{4,NSEEDS,T,NTANGENTS}
    neighbors::NTuple{NV,Tuple{CellNeighboring,Int}}
end

cell_boundary_polygon(cell::CoarsePolyCell) = cell.boundary

function update_dtype(::Type{CoarsePolyCell{T,NV,NS,NT}}) where {T,NV,NS,NT}
    return Tuple{SVector{4,T},SVector{4,T},SMatrix{4,NS,T,NT},SMatrix{4,NS,T,NT}}
end

function compute_coarse_cell_contents(
    empty_cell::CoarsePolyCell{T,NV,NSEEDS,NTANGENTS},
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NSEEDS,NTANGENTS}},
    tstep,
    boundary_conditions,
    gas,
) where {T,NV,NSEEDS,NTANGENTS}
    big_poly = cell_boundary_polygon(empty_cell)
    A, dA = DifferentiationInterface.value_and_gradient(
        poly_area,
        fdiff_backend,
        PlanePolygons._flatten(big_poly),
    )

    _, cells = nth_step(sim, tstep)
    min_cell_size = minimum_cell_size(sim)
    overlapped = all_cells_overlapping(big_poly, sim; padding = min_cell_size)
    U_total, Udot_total =
        total_mass_contained_by(big_poly, sim, tstep; poly_bbox_padding = min_cell_size)

    dMdpts = mapreduce(+, overlapped) do id
        p_cell = cell_boundary_polygon(cells[id])
        p_union = poly_intersection(big_poly, p_cell)
        A_union = poly_area(p_union)
        du_dx_vtxs = mapreduce(hcat, edge_starts(p_union)) do pt
            ∇u_at(sim, tstep, pt..., boundary_conditions, gas; padding = 2 .* min_cell_size)
        end
        dxj = mapreduce(vcat, edge_starts(p_union)) do pt
            intersection_point_jacobian(pt, big_poly, p_union)
        end
        dAdxj = intersection_area_jacobian(PlanePolygons._flatten(big_poly), p_cell)
        return cells[id].u * dAdxj' + A_union * du_dx_vtxs * dxj
    end
    du_dx_vtxs = (A * dMdpts - U_total * dA') / (A * A)
    return U_total / A, Udot_total / A, du_dx_vtxs
end

end
