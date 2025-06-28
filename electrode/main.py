import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace
from dolfinx.fem import petsc  # Direct import of petsc submodule
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

# Physical constants
k = 0.5             # Thermal conductivity of brain tissue [W/m·K]
rho_c = 3.8e6       # Volumetric heat capacity [J/m³·K]
alpha = k / rho_c   # Thermal diffusivity [m²/s]
Q_value = 1e6       # Heat generation [W/m³]

L = 0.01  # 1 cm domain
nx, ny = 50, 50  # Mesh resolution

domain = mesh.create_rectangle(MPI.COMM_WORLD,
                               [np.array([0, 0]), np.array([L, L])],
                               [nx, ny],
                               mesh.CellType.triangle)

# with XDMFFile(domain.comm, "mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)

# Create function space using the simpler approach for FEniCSx 0.9
V = functionspace(domain, ("Lagrange", 1))

T_init = fem.Function(V)
T_init.x.array[:] = 37.0  # Initial temp (Celsius)

def boundary(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0),
        np.isclose(x[0], L),
        np.isclose(x[1], 0),
        np.isclose(x[1], L)
    ])

bc_dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(37.0)), bc_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Tn = fem.Function(V)  # Previous timestep
Tn.x.array[:] = 37.0

dt = 0.01  # Smaller timestep for better accuracy

# Point heat source near center (made broader and stronger)
x = ufl.SpatialCoordinate(domain)
# Increased the standard deviation to make heat source broader
Q_expr = Q_value * ufl.exp(-((x[0]-L/2)**2 + (x[1]-L/2)**2)/(1e-6))  # Changed from 1e-7 to 1e-6

F = (u - Tn) / dt * v * ufl.dx + alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    - Q_expr / rho_c * v * ufl.dx

# Extract bilinear and linear forms
a = ufl.lhs(F)
L = ufl.rhs(F)

# Create solution function  
T = fem.Function(V)
steps = 1000  # More time steps to see heat diffusion

with io.XDMFFile(domain.comm, "results.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    T.name = "Temperature"

    for n in range(steps):
        # Assemble and solve using the correct FEniCSx 0.9 API
        A = petsc.assemble_matrix(fem.form(a), bcs=[bc])
        A.assemble()
        b = petsc.assemble_vector(fem.form(L))
        petsc.apply_lifting(b, [fem.form(a)], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.CG)
        solver.getPC().setType(PETSc.PC.Type.NONE)
        solver.solve(b, T.x.petsc_vec)
        T.x.scatter_forward()
        
        Tn.x.array[:] = T.x.array
        if n % 10 == 0:
            print(f"Step {n}: max T = {T.x.array.max():.2f} °C")
        xdmf.write_function(T, n * dt)