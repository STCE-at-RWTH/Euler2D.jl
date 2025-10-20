abstract type Obstacle end

struct CircularObstacle{T} <: Obstacle
    center::SVector{2,T}
    radius::T
end

function CircularObstacle(center, radius)
    CircularObstacle(SVector{2}(center...), radius)
end

function PlanePolygons.point_inside(s::CircularObstacle, pt)
    Δr = pt - s.center
    return sum(x -> x^2, Δr) <= s.radius^2
end

struct RectangularObstacle{T} <: Obstacle
    center::SVector{2,T}
    extent::SVector{2,T}
end

function PlanePolygons.point_inside(s::RectangularObstacle, pt)
    Δx = pt - s.center
    return all(abs.(Δx) .<= s.extent / 2)
end

struct ConvexPolygonalObstacle{N,T} <: Obstacle
    boundary::SClosedPolygon{N,T}
end

function ConvexPolygonalObstacle(pts...)
    return ConvexPolygonalObstacle(SClosedPolygon(pts...))
end

TriangularObstacle{T} = ConvexPolygonalObstacle{3,T}

function PlanePolygons.point_inside(s::ConvexPolygonalObstacle, pt)
    return point_inside(s.boundary, pt)
end
