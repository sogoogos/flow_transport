## Modeling flow and transport in porous media

This code is designed to solve the following flow (parabolic) and transport (hyperbolic) equations
with the finite volume method.

### Governing equations:
* Single phase flow in porous media:\
∂(φρ)/∂t +  ∇·(ρu) = q, \
where u is Darcy velocity: \
u = -k/μ ∇Φ. \
We often assume the fluid is incompressible or slightly compressible. 

* Transport equation: \
∂(φρc)/∂t +  ∇·(cρu) = 0.

where φ: porosity, ρ: fluid density, u: Darcy flux, q: sink/source, k: permeability tensor, μ: fluid viscosity,
Φ: potential, c: concentration of the component.


### Note
1. Currently, only 2 (or 1)-dimensional rectangular domains and so grids can be handled (3-D, unstructured grids are not implemented yet).
2. You should be able to solve other types of equations (e.g. heat equation, wave equation, etc.) in the same framework
 while the code is intended to solve flow and transport equations in porous media.
3. I wish Github markdown read Latex math.





