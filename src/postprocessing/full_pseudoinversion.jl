module FullPseudoinversion

using LinearAlgebra

using Accessors
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Graphs
using MetaGraphsNext
using OhMyThreads: tmap, tmapreduce
using StaticArrays
using Tullio

using Euler2D
using Euler2D: CellNeighboring
using Euler2D: shift_polygon_vertices_in_direction, fdiff_eps, select_middle
using Euler2D: all_cells_overlapping, total_mass_contained_by
using Euler2D: total_mass_contained_by_with_gradient_estimates
using Euler2D: fdiff_backend, estimate_∇u_in_rectangular_cell

using PlanePolygons

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

function intersection_point_jacobian(
    point,
    poly1::SizedClockwiseOrientedPolygon{N,T},
    poly2,
) where {N,T}
    J = @SArray zeros(T, 2, 2 * N)
    for (i, (p1, p2)) ∈ enumerate(zip(edge_starts(poly1), edge_ends(poly1)))
        if !is_other_point_on_line(Line(p1, p2 - p1), point)
            continue
        end
        for ℓ2 ∈ Iterators.filter(ℓ -> is_other_point_on_line(ℓ, point), edge_lines(poly2))
            jac = DifferentiationInterface.jacobian(fdiff_backend, vcat(p1, p2)) do vals
                q1 = vals[SVector(1, 2)]
                q2 = vals[SVector(3, 4)]
                return line_intersect(Line(q1, q2 - q1), ℓ2)
            end
            j = ((i + 1) % N) + 1
            @reset J[2*i-1:2*i] = jac[SVector(1, 2)]
            @reset J[2*j-1:2*j] = jac[SVector(3, 4)]
        end
    end
    return J
end

function intersection_point_gradient(
    point,
    poly1::SClosedPolygon{N,T},
    shift_dirs,
    poly2,
) where {N,T}
    @assert N == length(shift_dirs)
    J = @SArray zeros(T, 2, N)
    # A and B are on i-th edge of poly1
    for (i, (A, B)) ∈ enumerate(zip(edge_starts(poly1), edge_ends(poly1), shift_dirs))
        ℓ_AB = Line(A, B - A)
        if !is_other_point_on_line(ℓ_AB, point)
            continue
        end
        # the jacobian actually ends up being either all zeros on only having two nonzero columns?
        for ℓ_BC ∈ edge_lines(poly2)
            if !is_other_point_on_line(ℓ_BC, point)
                continue
            end
            # which pair of shifts i,j correspond to A, B
            j = ((i + 1) % N) + 1
            jac_local = DifferentiationInterface.jacobian(
                fdiff_backend,
                @SVector(zeros(T, 2)),
                Constant(A),
                Constant(B),
                Constant(shift_dirs[i]),
                Constant(shift_dirs[j]),
                Constant(ℓ_BC),
            ) do s, pA, pB, dirA, dirB, ℓ_BC_noshift
                A_shifted = pA + s[1] * dirA
                B_shifted = pB + s[2] * dirB
                ℓ_AB_shifted = Line(A_shifted, B_shifted - A_shifted)
                return line_intersect(ℓ_AB_shifted, ℓ_BC_noshift)
            end
            @reset J[:, i] = jac_local[:, 1]
            @reset J[:, j] = jac_local[:, 2]
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

"""
    intersection_area_gradient(poly1, s_dirs, poly2)

Compute the gradient of the intersection area of `poly1` and `poly2` with respect to
shift magnitudes in the given shift directions. 

We assume that `poly1` is what is shifted and that `poly2` remains constant.

Uses finite differences. There is almost certainly an AD-based solution here...
however, we are interested in some amount of "smoothing", in case the intersection area is not very large.
"""
function intersection_area_gradient(poly1::SClosedPolygon{N,T}, s_dirs, poly2) where {N,T}
    grad = @SVector zeros(T, N)
    h = fdiff_eps(zero(T))
    s = @SVector zeros(T, N)
    for i in eachindex(s, grad)
        poly1_minus_h = shift_polygon_vertices_in_direction(poly1, s_dirs, @set s[i] -= h)
        poly1_plus_h = shift_polygon_vertices_in_direction(poly1, s_dirs, @set s[i] += h)
        # maybe save a buffer here?
        A_minus = poly_area(poly_intersection(poly1_minus_h, poly2))
        A_plus = poly_area(poly_intersection(poly1_plus_h, poly2))
        @reset grad[i] = (A_plus - A_minus) / (2 * h)
    end
    return grad
end

## END POLYGON STUFF
struct PseudoinversionCell{T,NSEEDS,NTANGENTS} <: FVMCell{T}
    id::Int
    boundary::SClosedPolygon{4,T}
    vertex_shift_dirs::SVector{4,SVector{2,T}}
    u::SVector{4,T}
    u̇::SMatrix{4,NSEEDS,T,NTANGENTS}
    du_ds::SMatrix{4,4,T,16}
end

Euler2D.cell_boundary_polygon(cell::PseudoinversionCell) = cell.boundary

using Euler2D.ShockInterpolation: n_points, x_knots, y_knots
using Euler2D.ShockInterpolation: interpolation_segments, interpolation_normals_at_points

"""
    pseudoinversion_cell_polys(bow_shock_interp, L)

Build the set of polygons for pseudoinversion with tangent-to-shock side lengths `L`.
"""
function pseudoinversion_cell_polys(bow_shock_interp, L)
    segments = interpolation_segments(bow_shock_interp)
    corner_normals = interpolation_normals_at_points(bow_shock_interp)

    out = SClosedPolygon{4,Float64}[]
    sizehint!(out, 3 * n_points(bow_shock_interp))
    for i = 2:n_points(bow_shock_interp)
        n1 = corner_normals[i-1]
        n2 = corner_normals[i]
        A, B = segments[i]
        Qi_L = SClosedPolygon(A, A + L * n1, B + L * n2, B)
        Qi = SClosedPolygon(A, B, B - L * n2, A - L * n1)
        Qi_R = SClosedPolygon(A - L * n1, B - L * n2, B - 2 * L * n2, A - 2 * L * n2)
        push!(out, Qi_L, Qi, Qi_R)
    end

    return out
end

# Page 23 of my red notebook; 
function _du_ds_via_product_rule(
    u_j,
    ∇x_u_j,
    x_center_j,
    sim_cell_boundary,
    pinv_cell_boundary,
    pinv_shift_dirs,
)
    # intersection between the pinv and simulation cells
    p_isect = poly_intersection(pinv_cell_boundary, sim_cell_boundary)
    # area of that intersection
    A_isect = poly_area(p_isect)
    # and its dependence on s
    dA_isect_ds =
        intersection_area_gradient(pinv_cell_boundary, pinv_shift_dirs, sim_cell_boundary)

    # Approximate value of the linear interpolation of u in the intersection...
    u_isect = inv(num_vertices(p_isect)) * sum(edge_starts(p_isect)) do pt_l
        return u_j + ∇x_u_j * (pt_l - x_center_j)
    end
    # dependence of u_isect on the shifts s
    du_isect_ds =
        inv(num_vertices(p_isect)) *
        ∇x_u_j *
        sum(edge_starts(p_isect)) do pt_l
            ∇s_pt_l = intersection_point_gradient(
                pt_l,
                pinv_cell_boundary,
                pinv_shift_dirs,
                sim_cell_boundary,
            )
            # factor out ∇u, since it's constant (linear approximation)
            return ∇s_pt_l
        end
    # A is a scalar -> dAds is a row vector
    # (u in R^4) * (dA' in R^[1 x n_s]) * Δs -> something in R^4
    # plus
    # (A in R) * (du_ds in R^[4 x n_s]) -> something in R^4
    # sizes check out
    return u_isect * dA_isect_ds' + A_isect * du_isect_ds
end

function compute_pseudoinversion_cell_contents(
    pinv_cell::PseudoinversionCell{T,NS,NT},
    sim::CellBasedEulerSim{T,TangentQuadCell{T,NS,NT}},
    tstep,
    boundary_conditions,
    gas,
) where {T,NS,NT}
    pinv_cell_boundary = cell_boundary_polygon(pinv_cell)
    A, dAds = DifferentiationInterface.value_and_gradient(
        fdiff_backend,
        @SVector(zeros(T, 4)),
        Constant(pinv_cell_boundary),
        Constant(pinv_cell.vertex_shift_dirs),
    ) do s, poly, ns
        return poly_area(shift_polygon_vertices_in_direction(poly, ns, s))
    end
    _, cells = nth_step(sim, tstep)
    min_cell_size = minimum_cell_size(sim)
    overlapped = all_cells_overlapping(pinv_cell_boundary, sim; padding = min_cell_size)
    U_tot, Udot_tot = total_mass_contained_by_with_gradient_estimates(
        pinv_cell_boundary,
        sim,
        tstep,
        boundary_conditions,
        gas;
        poly_bbox_padding = min_cell_size,
    )
    # compute the first term of the product rule expansion of
    # partial u partial s for the pinv cell
    # the entire derivation is on page 22 of my red notebook
    # eqn is marked with **
    # Interested in the first sum term inside the big brackets
    # namely Σ_j ∂_s_k * ∬_(Cj ∩ Qi) u(x) dA
    dU_tot_ds = mapreduce(+, overlapped) do id
        ∇_x_u = estimate_∇u_in_rectangular_cell(cells[id], sim, tstep, boundary_conditions, gas)
        return _du_ds_via_product_rule(
            cells[id].u,
            ∇_x_u,
            cell_center(cells[id]),
            cell_boundary_polygon(cells[id]),
            pinv_cell_boundary,
            pinv_cell.vertex_shift_dirs,
        )
    end
    du_ds = inv(A^2) * (A * dU_tot_ds - U_tot * dAds')
    return inv(A) * U_tot, inv(A) * Udot_tot, du_ds
end

end
