module LocalPseudoinversion

using Accessors
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Graphs
using LinearAlgebra
using MetaGraphsNext
using OhMyThreads: tmap, tmapreduce
using Tullio
using StaticArrays

using Euler2D
using Euler2D: CellNeighboring, fdiff_eps, select_middle
using Euler2D: all_cells_overlapping, total_mass_contained_by
using PlanePolygons

const fdiff_backend = AutoForwardDiff()

### SOME POLYGON STUFF
# To my future self: sorry

"""
    intersection_point_jacobian(point, poly1, poly2)

Get the Jacobian of the point (x, y) w.r.t the vertices of `poly1`,
provided (x, y) is the intersection of a pair of edges of `poly1` and `poly2`.
"""
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
# allocates like hell, but... it works and isn't performance critical.?
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

## END POLYGON STUFF

## BOW-SHOCK PROBLEM SPECIFIC

"""
    CoarsePolyCell{T, NV}
"""
struct CoarseQuadCell{T,NSEEDS,NTANGENTS} <: FVMCell{T}
    id::Int
    boundary::SClosedPolygon{4,T}
    u::SVector{4,T}
    u̇::SMatrix{4,NSEEDS,T,NTANGENTS}
    #TODO this is a problem if NV is not 4!
    du_dvtxs::SMatrix{4,8,T,32}
end

Euler2D.cell_boundary_polygon(cell::CoarseQuadCell) = cell.boundary

function Euler2D.update_dtype(::Type{CoarseQuadCell{T,NS,NT}}) where {T,NS,NT}
    return Tuple{SVector{4,T},SVector{4,T},SMatrix{4,NS,T,NT},SMatrix{4,NS,T,NT}}
end

function coarse_cell_neighbor_data(c1, c2)
    if (c1.id == c2.id)
        return (false, nothing)
    end
    e1 = cell_boundary_polygon(c1)
    e2 = cell_boundary_polygon(c2)
    # TODO remove hardcoded direction data please
    dirs = (:south, :west, :north, :east)
    for (dir, p1, p2) ∈ zip(dirs, edge_starts(e1), edge_ends(e1))
        for (q1, q2) ∈ zip(edge_ends(e2), edge_starts(e2))
            if (
                are_points_collinear_between(p1, p2, q1, q2) ||
                are_points_collinear_between(q1, q2, p1, p2)
            )
                return (true, (dir, p1, p2))
            end
        end
    end
    return (false, nothing)
end

function compute_coarse_cell_contents(
    coarse_cell_boundary,
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NSEEDS,NTANGENTS}},
    tstep,
    boundary_conditions,
    gas,
) where {T,NSEEDS,NTANGENTS}
    A, dA = DifferentiationInterface.value_and_gradient(
        poly_area,
        fdiff_backend,
        PlanePolygons._flatten(coarse_cell_boundary),
    )
    # @show A, dA

    _, cells = nth_step(sim, tstep)
    min_cell_size = minimum_cell_size(sim)
    overlapped = all_cells_overlapping(coarse_cell_boundary, sim; padding = min_cell_size)
    U_total, Udot_total = total_mass_contained_by(
        coarse_cell_boundary,
        sim,
        tstep;
        poly_bbox_padding = min_cell_size,
    )

    # easy speedup to be had by preallocating for the gradient calls
    dU_totaldpts = mapreduce(+, overlapped) do id
        u_loc = cells[id].u
        p_cell = cell_boundary_polygon(cells[id])
        p_union = poly_intersection(coarse_cell_boundary, p_cell)
        # dependence of the overlapping area on its corners
        # A_union, dA_union_dxj = DifferentiationInterface.value_and_gradient(
        #     poly_area,
        #     fdiff_backend,
        #     PlanePolygons._flatten(p_union),
        # )
        A_union = poly_area(p_union)
        dA_union_dxj = intersection_area_jacobian(
            PlanePolygons._flatten(coarse_cell_boundary),
            p_cell,
        )
        # # dependence of u(x) at each of the overlapping area's corners
        du_dx_vtxs = mapreduce(hcat, edge_starts(p_union)) do pt
            ∇u_at(sim, tstep, pt..., boundary_conditions, gas; padding = 2 .* min_cell_size)
        end
        # dependence of the overlapping area's corners on the big cell's corners
        dxj = mapreduce(vcat, edge_starts(p_union)) do pt
            intersection_point_jacobian(pt, coarse_cell_boundary, p_union)
        end
        return u_loc * dA_union_dxj' + A_union * du_dx_vtxs * dxj
    end
    # @show size(U_total), size(dU_totaldpts), size(dA)
    du_dx_vtxs = (A * dU_totaldpts - U_total * dA') / (A * A)
    return U_total / A, Udot_total / A, du_dx_vtxs
end

function create_empty_coarse_cell_polys(shock_interp, num_cells_y, cell_width; s_max = 1.0)
    num_pts_y = num_cells_y + 1
    s_range = range(; start = 0.0, stop = s_max, length = num_pts_y)
    shock_point(s) = begin
        if s < 0
            return SMatrix{2,2,Float64,4}(1.0, 0.0, 0.0, -1.0) * shock_interp(-s)
        else
            return shock_interp(s)
        end
    end
    Δs = step(s_range)
    all_s = -Δs:Δs:s_max
    Δx = Point(cell_width, 0.0)
    return mapreduce(vcat, zip(all_s[begin:end-1], all_s[begin+1:end])) do (s0, s1)
        return [
            SClosedPolygon(
                shock_point(s0) + i * Δx,
                shock_point(s0) + (i - 1) * Δx,
                shock_point(s1) + (i - 1) * Δx,
                shock_point(s1) + i * Δx,
            ) for i ∈ 0:2
        ]
    end
end

function populate_coarse_cell(
    id,
    cell_poly,
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NSEEDS,NTANGENTS}},
    tstep,
    boundary_conditions,
    gas,
) where {T,NSEEDS,NTANGENTS}
    needs_flip = all(pt -> pt[2] ≤ 0.0, edge_starts(cell_poly))
    # TODO this assumes that there's always a StrongWall at y=0!
    # THIS WILL NOT WORK FOR THE EDNEY INTERACTIONS! BE WARNED!
    actual_poly = if needs_flip
        flipped = map(edge_starts(cell_poly)) do pt
            return SVector(pt[1], -pt[2])
        end
        SClosedPolygon(reverse(flipped)...)
    else
        cell_poly
    end
    u, udot, du_dx_vtxs_maybenans =
        compute_coarse_cell_contents(actual_poly, sim, tstep, boundary_conditions, gas)
    du_dx_vtxs = map(v -> isnan(v) ? zero(T) : v, du_dx_vtxs_maybenans)
    return CoarseQuadCell(id, cell_poly, u, udot, du_dx_vtxs)
end

struct DualNodeKind{S} end

const DUAL_GRAPH_NODE_TYPE = Tuple{
    DualNodeKind{S},
    Union{Nothing,CoarseQuadCell{T,NS,NTANGENTS},SVector{4,T}},
} where {S,T,NS,NTANGENTS}

const AMBIENT_BC_DUAL_IDX = 0

function populate_coarse_cell_graph(
    coarse_cells,
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NS,NTANGENTS}},
    free_stream_conditions,
    boundary_conditions,
) where {T,NS,NTANGENTS}
    g = MetaGraph(
        DiGraph(),
        Int,
        DUAL_GRAPH_NODE_TYPE{S,T,NS,NTANGENTS} where {S},
        Tuple{Symbol,Point{T},Point{T}},
    )
    for cell ∈ coarse_cells
        g[cell.id] = (DualNodeKind{:cell}(), cell)
    end
    g[AMBIENT_BC_DUAL_IDX] = (DualNodeKind{:boundary_ambient}(), free_stream_conditions)
    phantom_idx = 1000 * nv(g) + 1
    for cell ∈ coarse_cells
        for other ∈ coarse_cells
            flag, val = coarse_cell_neighbor_data(cell, other)
            if flag
                g[cell.id, other.id] = val
            end
        end
        dirs = (:south, :west, :north, :east)
        poly = cell_boundary_polygon(cell)
        for (p1, p2, dir, t, n) ∈ zip(
            edge_starts(poly),
            edge_ends(poly),
            dirs,
            edge_tangents(poly),
            outward_edge_normals(poly),
        )
            # TODO fix hardcoded directions and properties here
            if cell.id % 3 == 1 && dir == :west
                g[cell.id, AMBIENT_BC_DUAL_IDX] = (dir, p1, p2)
            end
        end
    end

    return g
end

function make_coarse_cell_graph(
    shock_interp,
    sim,
    tstep,
    free_stream_conditions,
    boundary_conditions,
    gas,
    num_cells_pos_y,
    cell_width;
    s_max = 1.0,
)
    empty_polys_enumerated = collect(
        enumerate(
            create_empty_coarse_cell_polys(
                shock_interp,
                num_cells_pos_y,
                cell_width;
                s_max = s_max,
            ),
        ),
    )
    cells_itr = tmap(empty_polys_enumerated) do (i, poly)
        populate_coarse_cell(i, poly, sim, tstep, boundary_conditions, gas)
    end
    return populate_coarse_cell_graph(
        cells_itr,
        sim,
        free_stream_conditions,
        boundary_conditions,
    )
end

function _marshal_edge_data(edge_data)
    return vcat(edge_data[2], edge_data[3])
end

function _unmarshal_edge_data(v)
    return (SVector(v[1], v[2]), SVector(v[3], v[4]))
end

function _marshal_edge_basis(args)
    return vcat(args...)
end

function _unmarshal_edge_basis(v)
    @assert length(v) == 7
    return (v[1], SVector(v[2], v[3]), SVector(v[4], v[5]), SVector(v[6], v[7]))
end

function _unmarshal_edge_data_gradient(g)
    @assert size(g) == (7, 4)
    return (g[1, :], g[SVector(2, 3), :], g[SVector(4, 5), :], g[SVector(6, 7), :])
end

function project_to_orthonormal_basis(v, new_basis_x)
    B = PlanePolygons.orthonormal_basis(new_basis_x)
    return change_basis(v, B)
end

function project_state_to_orthonormal_basis(u, new_basis_x)
    B = PlanePolygons.orthonormal_basis(new_basis_x)
    ρv_B = change_basis(Euler2D.select_middle(u), B)
    return SVector(u[begin], ρv_B..., u[end])
end

function _edge_basis(edge_data)
    (_, p1, p2) = edge_data
    L = norm(p2 - p1)
    t̂ = (p2 - p1) / L
    n̂ = SVector(-t̂[2], t̂[1])
    ê1 = project_to_orthonormal_basis(SVector(1.0, 0.0), n̂)
    return L, n̂, t̂, ê1
end

function _edge_basis_and_gradient(edge_data)
    blob = _marshal_edge_data(edge_data)
    res, dres = value_and_jacobian(fdiff_backend, blob) do v
        p1, p2 = _unmarshal_edge_data(v)
        return _marshal_edge_basis(_edge_basis((nothing, p1, p2)))
    end
    return (_unmarshal_edge_basis(res), _unmarshal_edge_data_gradient(dres))
end

# compute the "index" for the coarse dual component "nbr"
# because the values in nbr may actually be computed directly from u
# or from "ambient" (idx is 1)
_u_idx(::DualNodeKind{:boundary_sym}, id, nbr) = id
_u_idx(::DualNodeKind{:boundary_vN}, id, nbr) = id
_u_idx(::DualNodeKind{:boundary_body}, id, nbr) = id
_u_idx(::DualNodeKind{:cell}, id, nbr) = nbr
_u_idx(::DualNodeKind{:boundary_ambient}, id, nbr) = 1

function _compute_ϕ(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:cell},
    other_data,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
    u_R = project_state_to_orthonormal_basis(other_data.u, n̂)
    ϕ_n = ϕ(u_L, u_R, 1, gas)
    return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
end

function _compute_ϕ(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:boundary_vN},
    ::Nothing,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
    u_R = project_state_to_orthonormal_basis(cell_data.u, n̂)
    ϕ_n = ϕ(u_L, u_R, 1, gas)
    return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
end

function _compute_ϕ(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:boundary_ambient},
    other_data,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
    u_R = project_state_to_orthonormal_basis(other_data, n̂)
    ϕ_n = ϕ(u_L, u_R, 1, gas)
    return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
end

function _compute_ϕ(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::Union{DualNodeKind{:boundary_sym},DualNodeKind{:boundary_body}},
    ::Nothing,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
    ρv = select_middle(cell_data.u)
    ρv_reflected = -(ρv ⋅ n̂) * n̂ + (ρv ⋅ t̂) * t̂
    other_u = SVector(cell_data.u[1], ρv_reflected..., cell_data.u[4])
    u_R = project_state_to_orthonormal_basis(other_u, n̂)
    ϕ_n = ϕ(u_L, u_R, 1, gas)
    return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
end

function _compute_grad_ϕ_edge(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:cell},
    other_data,
    edge_data,
    gas,
)
    blob = _marshal_edge_data(edge_data)
    dϕ_dedge = jacobian(
        fdiff_backend,
        blob,
        Constant(cell_data.u),
        Constant(other_data.u),
    ) do v, cell_u, other_u
        p1, p2 = _unmarshal_edge_data(v)
        (L, n̂, t̂, ê1) = _edge_basis((nothing, p1, p2))
        u_L = project_state_to_orthonormal_basis(cell_u, n̂)
        u_R = project_state_to_orthonormal_basis(other_u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end
    return dϕ_dedge
end

function _compute_grad_ϕ_edge(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::Union{DualNodeKind{:boundary_sym},DualNodeKind{:boundary_body}},
    ::Nothing,
    edge_data,
    gas,
)
    blob = _marshal_edge_data(edge_data)
    dϕ_dedge = jacobian(fdiff_backend, blob, Constant(cell_data.u)) do v, u
        p1, p2 = _unmarshal_edge_data(v)
        (L, n̂, t̂, ê1) = _edge_basis((nothing, p1, p2))
        u_L = project_state_to_orthonormal_basis(u, n̂)
        ρv = select_middle(u)
        ρv_reflected = -(ρv ⋅ n̂) * n̂ + (ρv ⋅ t̂) * t̂
        other_u = SVector(u[1], ρv_reflected..., u[4])
        u_R = project_state_to_orthonormal_basis(other_u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end
    return dϕ_dedge
end

function _compute_grad_ϕ_u(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:cell},
    other_data,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    dϕ_dcell = jacobian(fdiff_backend, cell_data.u, Constant(other_data.u)) do u, u_other
        u_L = project_state_to_orthonormal_basis(u, n̂)
        u_R = project_state_to_orthonormal_basis(u_other, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    dϕ_dother = jacobian(fdiff_backend, other_data.u) do u
        u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
        u_R = project_state_to_orthonormal_basis(u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    return dϕ_dcell, dϕ_dother
end

function _compute_grad_ϕ_u(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:boundary_ambient},
    other_data,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
        u_L = project_state_to_orthonormal_basis(u, n̂)
        u_R = project_state_to_orthonormal_basis(other_data, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    dϕ_dother = jacobian(fdiff_backend, other_data) do u
        u_L = project_state_to_orthonormal_basis(cell_data.u, n̂)
        u_R = project_state_to_orthonormal_basis(u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end

    return dϕ_dcell, dϕ_dother
end

function _compute_grad_ϕ_u(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::Union{DualNodeKind{:boundary_sym},DualNodeKind{:boundary_body}},
    ::Nothing,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
        u_L = project_state_to_orthonormal_basis(u, n̂)
        ρv = select_middle(u)
        ρv_reflected = -(ρv ⋅ n̂) * n̂ + (ρv ⋅ t̂) * t̂
        other_u = SVector(u[1], ρv_reflected..., u[4])
        u_R = project_state_to_orthonormal_basis(other_u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end
    return dϕ_dcell, zero(dϕ_dcell)
end

function _compute_grad_ϕ_u(
    ϕ,
    ::DualNodeKind{:cell},
    cell_data,
    ::DualNodeKind{:boundary_vN},
    ::Nothing,
    edge_data,
    gas,
)
    (L, n̂, t̂, ê1) = _edge_basis(edge_data)
    dϕ_dcell = jacobian(fdiff_backend, cell_data.u) do u
        u_L = project_state_to_orthonormal_basis(u, n̂)
        u_R = project_state_to_orthonormal_basis(u, n̂)
        ϕ_n = ϕ(u_L, u_R, 1, gas)
        return L * project_state_to_orthonormal_basis(ϕ_n, ê1)
    end
    return dϕ_dcell, zero(dϕ_dcell)
end

function boundary_integral(ϕ, dual, id)
    nbrs = neighbor_labels(dual, id)
    return sum(nbrs) do nbr
        return _compute_ϕ(ϕ, dual[id]..., dual[nbr]..., dual[id, nbr])
    end
end

function boundary_integral_gradient(ϕ, dual, id, gas)
    edge_gradients = map(neighbor_labels(dual, id)) do nbr
        cell_kind, cell_data = dual[id]
        nbr_kind, nbr_data = dual[nbr]
        edge_data = dual[id, nbr]
        edge_length = norm(edge_data[3] - edge_data[2])
        dϕ_did, dϕ_dnbr = _compute_grad_ϕ_u(
            ϕ,
            cell_kind,
            cell_data,
            nbr_kind,
            nbr_data,
            edge_data,
            gas,
        )
        dLϕ_dedge = _compute_grad_ϕ_edge(
            ϕ,
            cell_kind,
            cell_data,
            nbr_kind,
            nbr_data,
            edge_data,
            gas,
        )
        other_idx = _u_idx(nbr_kind, id, nbr)
        return (edge_data[1], id, other_idx, dϕ_did, dϕ_dnbr, edge_length, dLϕ_dedge)
    end
    ddparams = sum(edge_gradients) do (_, i, j, dϕ_dui, dϕ_duj, _, _)
        return dϕ_dui * dual[i][2].u̇ + dϕ_duj * dual[j][2].u̇
    end
    _edge_point_indices = (north = (3, 4), south = (1, 2), east = (4, 1), west = (2, 3))
    _edge_opposites = (north = :south, south = :north, east = :west, west = :east)
    ddpts = zeros(size(ddparams)[1], 8)
    foreach(edge_gradients) do (dir, i, j, dϕ_dui, dϕ_duj, L, dLϕ_dpts)
        local (k, l) = _edge_point_indices[dir]
        # reversed numbering in the cell on the other side!
        local (n, m) = _edge_point_indices[_edge_opposites[dir]]
        ddpts[:, (2*k-1):2*k] += dLϕ_dpts[:, SVector(1, 2)]
        ddpts[:, (2*l-1):2*l] += dLϕ_dpts[:, SVector(3, 4)]

        ddpts[:, (2*k-1):2*k] += dϕ_dui * dual[i][2].du_dvtxs[:, (2*k-1):2*k]
        ddpts[:, (2*l-1):2*l] += dϕ_dui * dual[i][2].du_dvtxs[:, (2*l-1):2*l]

        ddpts[:, (2*k-1):2*k] += dϕ_duj * dual[j][2].du_dvtxs[:, (2*m-1):2*m]
        ddpts[:, (2*l-1):2*l] += dϕ_duj * dual[j][2].du_dvtxs[:, (2*n-1):2*n]
    end
    return ddparams, ddpts
end

const x_select = hcat(
    SVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    SVector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
)

function ξ_local_pseudoinversion(ϕ, dual, id, gas) # where the magic happens
    # A = boundary_integral_ddparams(dual, id)
    # @assert rank(A) == min(size(A)...)
    # B = boundary_integral_ddL(dual, id)
    # @assert rank(B) == min(size(B)...)
    # C = pinv(B)*A
    local (A, B) = boundary_integral_gradient(ϕ, dual, id, gas)
    local C = pinv(B * x_select)
    return -(C * A)
end

end
