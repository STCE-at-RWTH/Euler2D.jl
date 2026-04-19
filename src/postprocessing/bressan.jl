# DIRECT EXTENSION OF THE BRESSAN CALCULUS
# TO THE BOW-SHOCK PROBLEM
#
# does it work? who can know such things

module Bressan

using LinearAlgebra

using Interpolations
using OhMyThreads
using StaticArrays

using Euler2D
using PlanePolygons

struct FieldInterpolationBressan{T, DV, NEPS, FIELD, DFIELD, REGION_T}
    cell_centers::Tuple{DV, DV}
    shock_aabb::SClosedPolygon{4, T}
    ε::SVector{T, NEPS}
    field::FIELD
    dfield::DFIELD
    shift_regions::Vector{REGION_T}
end

function poly_aabb(polys...)
    T = eltype(first(edge_starts(first(polys))))
    x1 = typemax(T)
    x2 = typemin(T)
    y1 = typemax(T)
    y2 = typemin(T)
    for pt in Iterators.flatmap(edge_starts, polys)
        x1 = min(x1, pt[1])
        x2 = max(x2, pt[1])
        y1 = min(y1, pt[2])
        y2 = max(y2, pt[2])
    end
    return SClosedPolygon(
        SVector(x1, y1),
        SVector(x1, y2),
        SVector(x2, y2),
        SVector(x2, y1),
    )
end

# right normal of first edge of each "region" points in the direction of the shift
function _untwist_shifted_shock(Γ1, Γ2, s1, s2)
    A = Γ2(s1)
    B = Γ1(s1)
    C = Γ1(s2)
    D = Γ2(s2)
    AB = PlanePolygons.Line(A, B - A)
    CD = PlanePolygons.Line(C, D - C)
    E = (A + B + C + D) / 4
    AB_r = point_in_right_half_plane(AB, E)
    CD_r = point_in_right_half_plane(CD, E)
    res = if AB_r && CD_r
        [SClosedPolygon(B, C, D, A)]
    elseif !AB_r && !CD_r
        [SClosedPolygon(C, B, A, D)]
    elseif AB_r && !CD_r
        [SClosedPolygon(B, E, A), SClosedPolygon(C, E, D)]
    else
        [SClosedPolygon(E, B, A), SClosedPolygon(E, C, D)]
    end
    return res
end

function shifted_regions(Γ1, Γ2, s)
    s_first = s[begin:(end - 1)]
    s_second = s[(begin + 1):end]
    return mapreduce((s1, s2) -> _untwist_shifted_shock(Γ1, Γ2, s1, s2), vcat, s_first, s_second)
end

end
