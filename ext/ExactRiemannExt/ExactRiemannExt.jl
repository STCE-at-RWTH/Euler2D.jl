module ExactRiemannExt

using NonlinearSolve
using StaticArrays

using Euler2D
using Euler2D.ExactRiemannSolverInternal: project_state_to_normal, jump_ratios
using Euler2D.ExactRiemannSolverInternal: f_1, f_3, h_1, h_3

function f(x, params)
    (p, gas) = params
    # 18.62 stuff - A = 0
    v1 = f_1(x[1], gas) * exp(x[2]) * f_3(x[3], gas) - p[1]
    # 18.61: stuff - B = 0 
    v2 = exp(x[3] - x[1]) - p[2]
    # 18.63: stuff - C = 0
    v3 = h_1(x[1], gas) + sqrt(p[2] / p[1]) * h_1(x[1] + log(p[2])) - p[3]
    return SVector(v1, v2, v3)
end

function Euler2D.ExactRiemannSolverInternal.solve_for_jump_parameters(
    state_left,
    state_right,
    gas,
    n = 1,
)
    uL = project_state_to_normal(state_left, n)
    uR = project_state_to_normal(state_right, n)

    p = SVector(jump_ratios(uL, uR, gas)...)
    nonlinear_system = NonlinearProblem(f, ones(uL), (p, gas))
    solution = solve(nonlinear_system)
    return solution.u
end

end
