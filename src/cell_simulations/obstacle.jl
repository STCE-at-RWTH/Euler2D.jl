abstract type Obstacle end

struct CircularObstacle{T} <: Obstacle
    center::SVector{2,T}
    radius::T
end

function CircularObstacle(center, radius)
    CircularObstacle(SVector{2}(center...), radius)
end

function point_inside(s::CircularObstacle, pt)
    Δr = pt - s.center
    return sum(x -> x^2, Δr) < s.radius^2
end

struct ParametricObstacle{T} <: Obstacle
    x_func::Function            # Function defining x(t)
    y_func::Function            # Function defining y(t)
    parameters::NamedTuple      # Parameters like a, b, h, k
    shape_type::Symbol          # :circle, :ellipse, :hyperbola or :polynomial
    center::Union{SVector{2, Float64}, Nothing}  # Center of the obstacle (h, k) or `nothing`
end

function ParametricObstacle(
    x_func::Function,
    y_func::Function,
    parameters::NamedTuple,
    shape_type::Symbol
)
    # Validate that shape_type is one of the allowed types
    @assert shape_type in (:circle, :ellipse, :hyperbola, :polynomial) "Invalid shape type: $shape_type"
    
    # Handle center based on shape_type
    center = if shape_type in (:circle, :ellipse, :hyperbola)
        # Extract and explicitly cast the center (h, k) for these shapes
        @assert :h in fieldnames(typeof(parameters)) && :k in fieldnames(typeof(parameters)) "Parameters must include :h and :k"
        SVector(parameters.h, parameters.k)
    elseif shape_type == :polynomial
        # No center for polynomials
        nothing
    else
        throw(ArgumentError("Unsupported shape type: $shape_type"))
    end

    # Return the obstacle
    return ParametricObstacle{typeof(x_func)}(x_func, y_func, parameters, shape_type, center)
end

function point_inside(obstacle::ParametricObstacle, point::SVector)
    if obstacle.shape_type == :circle
        # Check if the point is inside a circle
        dx = point[1] - obstacle.parameters.k
        dy = point[2] - obstacle.parameters.h
        return sqrt(dx^2 + dy^2) <= obstacle.parameters.r

    elseif obstacle.shape_type == :ellipse
        # Check if the point is inside an ellipse
        a = obstacle.parameters.a
        b = obstacle.parameters.b
        h = obstacle.parameters.h
        k = obstacle.parameters.k
        dx = (point[1] - h) / a
        dy = (point[2] - k) / b
        return dx^2 + dy^2 <= 1

    elseif obstacle.shape_type == :hyperbola
        # Check if the point satisfies the hyperbola equation (inside the curve)
        a = obstacle.parameters.a
        b = obstacle.parameters.b
        h = obstacle.parameters.h
        k = obstacle.parameters.k
        dx = (point[1] - h) / a
        dy = (point[2] - k) / b
        return dx^2 - dy^2 <= 1  # Hyperbola condition

    elseif obstacle.shape_type == :polynomial
        # Check if the point satisfies the polynomial curve
        a = obstacle.parameters.a
        b = obstacle.parameters.b
        c = obstacle.parameters.c
        #y_eval = a * point[1]^2 + b * point[1] + c
        _,_,t_to_eval,_,_,_,_ = from_cartesian_to_polar(point_A,point_A,obstacle.shape_type,obstacle.parameters)
        y_eval = obstacle.y_func(t_to_eval,obstacle.parameters)
        return (point[2] - y_eval) >= 0  # Allow small tolerance for precision issues

    else
        error("point_inside currently does not support shape type: $(obstacle.shape_type)")
    end
end

struct RectangularObstacle{T} <: Obstacle
    center::SVector{2,T}
    extent::SVector{2,T}
end

function point_inside(s::RectangularObstacle, pt)
    Δx = pt - s.center
    return all(abs.(Δx) .< s.extent / 2)
end

struct TriangularObstacle{T} <: Obstacle
    points::NTuple{3,SVector{2,T}}
end

"""
    TriangularObstacle(points)

Create a triangular obstacle from a clockwise-oriented list of its vertices.
"""
function TriangularObstacle(pts...)
    return TriangularObstacle(tuple((SVector{2}(p) for p ∈ pts)...))
end

function point_inside(s::TriangularObstacle, pt)
    return all(zip(s.points, s.points[[2, 3, 1]])) do (p1, p2)
        R = SMatrix{2,2}(0, -1, 1, 0)
        return (R * (p2 - p1)) ⋅ (pt - p1) > 0 # test if inward normal faces towards the point
    end
end
