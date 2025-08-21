```@meta
EditURL = "array_sims.jl"
```

# Array-based Euler Simulations

This tutorial will demonstrate how to run, save, and load one- and two-dimensional simluations of the
Euler equations on a regular grid. These simulations are done using a Godunov method.

## [1-Dimensional Simulation Script](@id array-1d-tutorial-script)

````@example array_sims
using Euler2D
using LinearAlgebra
using ShockwaveProperties
using Unitful

# we want to set up Sod's shock tube problem
ρL = 1.0u"kg/m^3"
vL = [0.0u"m/s", 0.0u"m/s"]
PL = 10.0u"Pa"
TL = uconvert(u"K", PL / (ρL * DRY_AIR.R))
ML = vL / speed_of_sound(ρL, PL, DRY_AIR)

ρR = 0.125 * ρL
vR = [0.0u"m/s", 0.0u"m/s"]
PR = 0.1 * PL
TR = uconvert(u"K", PR / (ρR * DRY_AIR.R))
MR = vR / speed_of_sound(ρR, PR, DRY_AIR)

s_high = ConservedProps(PrimitiveProps(ρL, [ML[1]], TL), DRY_AIR)
s_low = ConservedProps(PrimitiveProps(ρR, [MR[2]], TR), DRY_AIR)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

