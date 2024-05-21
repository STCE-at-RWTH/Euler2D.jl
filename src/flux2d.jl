## 2D Riemann problem for the euler equations.

"""
Named tuple to keep track of data in four cells that all share one vertex.
"""
CellData2D{T} = @NamedTuple begin
    NE::T
    NW::T
    SE::T
    SW::T
end

"""
Named tuple to keep track of interface data between four cells that share one vertex.
"""
InterfaceData2D{T} = @NamedTuple begin
    N::T
    S::T
    E::T
    W::T
end

"""
Named tuple to keep track of data in four cells that all share one vertex.
"""
IntermediateData2D{T} = @NamedTuple begin
    N★::T
    S★::T
    ★E::T
    ★W::T
end

# ideally, these methods would work with some iteration over the field names
function merge_data(a::CellData2D{T}, b::CellData2D{U})::CellData2D{Tuple{T,U}} where {T,U}
    (NE = (a.NE, b.NE), NW = (a.NW, b.NW), SE = (a.SE, b.SE), SW = (a.SW, b.SW))
end

function merge_data(
    a::CellData2D{T},
    b::CellData2D{U},
)::CellData2D{Tuple{fieldtypes(T...),U}} where {T<:Tuple,U}
    (NE = (a.NE..., b.NE), NW = (a.NW..., b.NW), SE = (a.SE..., b.SE), SW = (a.SW..., b.SW))
end

function merge_data(
    a::CellData2D{T},
    b::CellData2D{U},
)::CellData2D{Tuple{T,fieldtypes(U)...}} where {T,U<:Tuple}
    (NE = (a.NE, b.NE...), NW = (a.NW, b.NW...), SE = (a.SE, b.SE...), SW = (a.SW, b.SW...))
end

function merge_data(
    a::CellData2D{T},
    b::CellData2D{U},
)::CellData2D{Tuple{fieldtypes(T)...,fieldtypes(U)...}} where {T<:Tuple,U<:Tuple}
    (
        NE = (a.NE..., b.NE...),
        NW = (a.NW..., b.NW...),
        SE = (a.SE..., b.SE...),
        SW = (a.SW..., b.SW...),
    )
end

"""
Evaluates ``f(u_L, u_R, dim)`` across each interface in the 2-dimensional Riemann problem. 
"""
function evaluate_across_ifaces(f, data::CellData2D)
    const _cell_ifaces =
        (N = (:NE, :NW, 1), S = (:SE, :SW, 1), E = (:SE, :NE, 2), W = (:SW, :NE, 2))
    return map(_cell_ifaces) do left, right, dim
        f(getproperty(data, left), getproperty(data, right), dim)
    end
end

function interface_signal_speeds(data::CellData2D{T}; gas::CaloricallyPerfectGas)
    return evaluate_across_ifaces(data) do uL, uR, dim
        interface_signal_speeds(uL, uR, dim; gas = gas)
    end
end

function interface_states(
    u::CellData2D,
    f::CellData2D,
    s::CellData2D;
    gas::CaloricallyPerfectGas,
) end

function interface_fluxes(u::CellData2D{T}; gas::CaloricallyPerfectGas) where {T}
    fluxes = map(u) do u
        F(u; gas = gas)
    end
    f = map(fluxes) do f
    end
end

function strong_interaction_speeds(iface_speeds)
    return map(iface_speeds) do sL, sR
        max(abs(sL), abs(sR))
    end
end

# eqns 2.48 from Vides / derived by Balsara

f_★ν(u_★ν, g_★ν) = [
    u_★ν[2],
    g_★ν[3] + (u_★ν[2]^2 - u_★ν[3]^2) / u_★ν[1],
    u_★ν[3] * u_★ν[2] / u_★ν[1],
    u_★ν[2]g_★ν[4] / u_★ν[3],
]

g_μ★(u_μ★, f_μ★) = [
    u_μ★[3],
    u_μ★[2] * u_μ★[3] / u_μ★[1],
    f_μ★[2] + (u_μ★[3]^2 - u_μ★[2]^2) / u_μ★[1],
    u_μ★[3] * f_μ★[4] / u_μ★[2],
]

"""
    u_★★
Compute average value in the strong interaction state.

**Equation 3.9 from Vides et al**
"""
function u_★★(s::InterfaceData2D) end

"""
Returns ``(s_e, s_w, s_n, s_s)``.
"""
function strong_interaction_state(Δt, s_e, s_w, s_n, s_s) end