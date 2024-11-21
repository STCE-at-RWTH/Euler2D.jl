# IMPLEMENTATION OF
# Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows  
# Takeshi R. Fujimoto ∗, Taro Kawasaki 1, Keiichi Kitamura
# Journal of Computational Physics, 396, pp. 264 - 279

_diff_op(T) = SVector{3, T}(one(T), zero(T), -one(T))
_avg_op(T) = SVector{3, T}(one(T), 2*one(T), one(T))

function convolve_sobel(matrix::AbstractMatrix{T}) where {T}
    Gx = _avg_op(T) * _diff_op(T)'
    Gy = _diff_op(T) * _avg_op(T)'
    new_size = size(matrix) .- 2
    outX = similar(matrix, new_size)
    outY = similar(matrix, new_size)
    @tullio outX[i,j] := Gx * matrix[i:i+2, j:j+2]
    @tullio outY[i,j] := Gy * matrix[i:i+2, j:j+2]
    return outX, outY
end

function discrete_gradient_direction(θ)
    if -π/8 ≤ θ < π/8
        return 0
    elseif π/8 ≤ θ < 3*π/8
        return π/4
    elseif 3*π/8 ≤ θ < π/2 || -π/2 ≤ θ < -3π/8
        return π/2
    else
        return -π/4
    end
end

