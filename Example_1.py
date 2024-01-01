"""                                 ↑↑ V out m/s
                                    ↑↑ 
                                  | ↑↑ |
                                  | ↑↑ |
        ---------------------------    --------  ↑
        |                                     |  |
        |-> U_in                              |  |
        |                                     |  |
        |                                     |  |
        |                                     |  |
        |                                     |  |
        |                                     |  |
        |                                     |  | 1.0 m
        |                                     |  |
        |                                     |  |
        |                                     |  |
    -----                                     |  |
 --------> U in m/s                           |  |
    -------------------------------------------  ↓
        <------------ 1.2 m ------------>



"""
import fenics as fe
import dolfin as df
import numpy as np
from fenics import (
    Constant, FunctionSpace, TestFunctions, 
    Function, MixedElement, MeshFunction, 
    cells, refine, Measure, SubDomain, 
    derivative, NonlinearVariationalProblem, 
    NonlinearVariationalSolver, DirichletBC, split, 
    near, LogLevel, set_log_level, sqrt
)
from tqdm import tqdm
from mpi4py import MPI

set_log_level(LogLevel.ERROR)


########################## Tracking Information Functions and Dimenssionless Numbers ##################

def compute_global_velocity_extremes(upT, W, comm):
    """
    Compute the global maximum and minimum velocities across all MPI processes.

    Args:
    upT: dolfin.Function
        The current solution for velocity, pressure, and temperature.
    dm0: dolfin.DofMap
        Degree of freedom mapping for the velocity function space.
    comm: MPI communicator
        The MPI communicator for the simulation.

    Returns:
    tuple: A tuple containing the global maximum and minimum velocities.
    """
    # Define the dofmap for velocity
    dm0 = W.sub(0).dofmap()

    # Compute local max and min velocities
    u_max_local = np.abs(upT.vector().vec()[dm0.dofs()]).max()
    u_min_local = np.abs(upT.vector().vec()[dm0.dofs()]).min()

    # Compute global max and min velocities
    u_max = comm.allreduce(u_max_local, op=MPI.MAX)
    u_min = comm.allreduce(u_min_local, op=MPI.MIN)

    return u_max, u_min

def calculate_dimensionless_numbers(u_max, domain_length_x, K1, RHO1, MU1, grid_spacing_x ):
    """
    Calculate the Peclet and Reynolds numbers.

    Args:
    u_max: float
        The maximum velocity in the domain.
    Nx: float
        Characteristic length scale (e.g., domain size).
    K1: float
        Thermal conductivity of the fluid.
    RHO1: float
        Density of the fluid.
    MU1: float
        Dynamic viscosity of the fluid.

    Returns:
    tuple: A tuple containing the Peclet and Reynolds numbers.
    """
    # Calculate Peclet number (Advective/Diffusive transport rate)
    peclet_number = (u_max * domain_length_x) / K1

    # Calculate Reynolds number
    reynolds_number = RHO1 * u_max * domain_length_x / MU1

    CFL_condition = grid_spacing_x / u_max

    return peclet_number, reynolds_number, CFL_condition

#############################  END  ################################

#################### Define Parallel Variables ####################


# Get the global communicator
comm = MPI.COMM_WORLD

# Get the rank of the process
rank = comm.Get_rank()

# Get the size of the communicator (total number of processes)
size = comm.Get_size()

#############################  END  ################################


##################### Physical Constants ################################
    
GRAVITY = -10  # Acceleration due to gravity (m/s^2)
RHO1 = 1000  # Fluid density (kg/m^3)
MU1 = 10.0 # Dynamic viscosity (Pa.s)
K1 = 41800  # Thermal conductivity (W/m.K)
CP1 = 4184  # Heat capacity (J/kg.K)
ALPHA1 = 1.3 * 10**-3  # Thermal expansion coefficient (1/K)
GAMMA = -8 * 10 ** -5  # Surface tension temperature derivative (N/m.K)


# rho = 1000  # kg/m^3
# C_p = 4184  # J/kg

# # Effective Turbulent Properties
# k_t = 41800  # W/mK
# mu_t = 10.0  # kg/m/s

# Problem Constants:

top_outflow_start = 0.7   # m
top_outflow_end = 0.8   # m
v_top = 2.5     # m/s 
inflow_start = 0.0    # m
inflow_end = 0.2    # m
u_in = 1    # m/s

#############################  END  ################################

##################### Mesh Refinement Functions For Bounderies ######################

def refine_mesh_near_boundary(mesh, threshold, domain):
    """
    Refines the mesh near the boundaries based on a specified threshold.

    Parameters:
    mesh : dolfin.Mesh
        The initial mesh to be refined.
    threshold : float
        The distance from the boundaries where the mesh should be refined.
    domain : List of tuples
        Domain boundaries specified as [(X0, Y0), (X1, Y1)]
        where (X0, Y0) is the bottom-left and (X1, Y1) is the top-right corner.

    Returns:
    mesh_r : dolfin.Mesh
        The refined mesh.
    """
    
    # Unpack domain coordinates
    (X0, Y0), (X1, Y1) = domain
    
    # Initialize a MeshFunction for marking cells to refine
    marker = MeshFunction("bool", mesh, mesh.topology().dim(), False)

    # Iterate through each cell in the mesh
    for idx, cell in enumerate(cells(mesh)):
        x_mid, y_mid = cell.midpoint().x(), cell.midpoint().y()

        # Calculate the distance from the cell's midpoint to the boundary
        dist_to_left_boundary = abs(x_mid - X0)
        dist_to_right_boundary = abs(x_mid - X1)
        dist_to_bottom_boundary = abs(y_mid - Y0)
        dist_to_top_boundary = abs(y_mid - Y1)

        # Mark cells for refinement if they're within the threshold distance from any boundary
        if (min(dist_to_left_boundary, dist_to_right_boundary) < threshold or
            min(dist_to_bottom_boundary, dist_to_top_boundary) < threshold):
            marker.array()[idx] = True

    # Refine the mesh based on the marked cells
    refined_mesh = refine(mesh, marker)

    return refined_mesh


def refine_mesh_near_corners(mesh, threshold, domain):
    """
    Refines the mesh near the corners based on a specified threshold.

    Parameters:
    mesh : dolfin.Mesh
        The initial mesh to be refined.
    threshold : float
        The distance from the corners where the mesh should be refined.
    domain : List of tuples
        Domain boundaries specified as [(X0, Y0), (X1, Y1)]
        where (X0, Y0) is the bottom-left and (X1, Y1) is the top-right corner.

    Returns:
    refined_mesh : dolfin.Mesh
        The refined mesh near the corners.
    """
    
    # Unpack domain coordinates
    (X0, Y0), (X1, Y1) = domain
    
    # Initialize a MeshFunction for marking cells to refine
    marker = MeshFunction("bool", mesh, mesh.topology().dim(), False)

    # Iterate through each cell in the mesh
    for idx, cell in enumerate(cells(mesh)):
        x_mid, y_mid = cell.midpoint().x(), cell.midpoint().y()

        # Calculate the distance from the cell's midpoint to the corners
        dist_to_bottom_left_corner = sqrt((x_mid - X0)**2 + (y_mid - Y0)**2)
        dist_to_bottom_right_corner = sqrt((x_mid - X1)**2 + (y_mid - Y0)**2)
        dist_to_top_left_corner = sqrt((x_mid - X0)**2 + (y_mid - Y1)**2)
        dist_to_top_right_corner = sqrt((x_mid - X1)**2 + (y_mid - Y1)**2)

        # Mark cells for refinement if they're within the threshold distance from any corner
        if ( dist_to_top_left_corner < threshold):
            marker.array()[idx] = True

    # Refine the mesh based on the marked cells
    refined_mesh = refine(mesh, marker)

    return refined_mesh


#############################  END  ################################


############################## Define domain sizes and discretization parameters ################################

# Define approximate lengths of the domain in x and y directions (meters)
approx_domain_length_x = 12e-1  # m
approx_domain_length_y = 1      # m

# Define grid spacing in x and y directions (meters)
grid_spacing_x = 5e-2  # m
grid_spacing_y = 5e-2  # m 

# Define time step for the simulation (arbitrary units)
# Based on CFL dt should be less than : dx/u_max 
dt = 0.04 * 0.9

# Calculate the number of divisions along each axis based on approximate domain size and grid spacing
num_divisions_x = int(approx_domain_length_x / grid_spacing_x)
num_divisions_y = int(approx_domain_length_y / grid_spacing_y)

# Adjust the domain length to ensure it is divisible by the grid spacing and slightly larger than the desired size
domain_length_x = (num_divisions_x + 1) * grid_spacing_x
domain_length_y = (num_divisions_y + 1) * grid_spacing_y

# Update the number of divisions to match the new domain length
num_divisions_x += 1
num_divisions_y += 1

# Define the origin point of the domain (bottom left corner)
origin = df.Point(0.0, 0.0)

# Calculate the top right corner based on the origin and adjusted domain lengths
top_right_corner = df.Point(origin.x() + domain_length_x, origin.y() + domain_length_y)

# Create the initial rectangular mesh using the defined corners and number of divisions
initial_mesh = fe.RectangleMesh(origin, top_right_corner, num_divisions_x, num_divisions_y)

# Define Domain 

Domain = [ ( 0.0 , 0.0 ) ,( 0.0 + domain_length_x , 0.0 + domain_length_y ) ]

#############################  END  ################################

############################ Modify Initial Mesh ######################

mesh = initial_mesh

mesh  = refine_mesh_near_boundary( mesh, 0.1, Domain )
mesh  = refine_mesh_near_boundary( mesh, 0.1, Domain )


mesh  = refine_mesh_near_corners( mesh, 0.1, Domain  )


#############################  END  ################################


######################################################################

def create_function_spaces(mesh):
    """
    Create function spaces, test functions, and functions for velocity, pressure, and temperature.

    Args:
    mesh : fenics.Mesh
        The computational mesh.

    Returns:
    tuple: A tuple containing the function spaces, test functions, and current and previous solutions.
    """

    # Define finite elements for velocity, pressure, and temperature
    P2 = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)  # Velocity
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Pressure 

    # Define mixed elements
    element = MixedElement([P2, P1])

    # Create a function space
    W = FunctionSpace(mesh, element)

    # Define test functions
    v_test, q_test = TestFunctions(W)

    # Define current and previous solutions
    upT = Function(W)  # Current solution
    upT0 = Function(W)  # Previous solution

    # Split functions to access individual components
    u_answer, p_answer = split(upT)  # Current solution
    u_prev, p_prev = split(upT0)  # Previous solution

    return W, v_test, q_test, upT, upT0, u_answer, p_answer, u_prev, p_prev

# Usage example:
# W, v_test, q_test, upT, upT0, u_answer, p_answer, u_prev, p_prev= create_function_spaces(mesh)

#############################  END  ################################


############################ Defining Equations ###########################

# Related Functions for defining equaions
def epsilon(u):  
    """
    Calculate the strain rate tensor for a given velocity field.

    Args:
    u : dolfin.Function
        The velocity field.

    Returns:
    dolfin.Expression
        The strain rate tensor.
    """
    return 0.5 * (fe.grad(u) + fe.grad(u).T)

def sigma(u, p, mu1):
    """
    Calculate the stress tensor for a given velocity field and pressure.

    Args:
    u : dolfin.Function
        The velocity field.
    p : dolfin.Function
        The pressure field.

    Returns:
    dolfin.Expression
        The stress tensor.
    """
    return 2 * mu1 * epsilon(u) - p * fe.Identity(len(u))

def Traction(T, n_v, gamma):
    """
    Calculate the traction on the boundary for a given temperature field.

    Args:
    T : dolfin.Function
        The temperature field.
    n_v : dolfin.Expression or dolfin.Constant
        The normal vector to the boundary.

    Returns:
    dolfin.Expression
        The traction vector.
    """
    return gamma * (fe.grad(T) - fe.dot(n_v, fe.grad(T)) * n_v)


# main equaions

def F1(u_answer, q_test, dt):
    """
    Define the weak form of the continuity equation for incompressible flow.

    Args:
    u_answer: dolfin.Function
        The current approximation of the velocity field in the mixed function space.
    q_test: dolfin.TestFunction
        The test function for pressure in the mixed function space.
    dt: float
        The time step for the transient simulation.

    Returns:
    ufl.Form
        The weak form of the continuity equation suitable for FEniCS assembly.
    """
    
    # The weak form of the continuity equation for incompressible flow is the integral of the 
    # product of the test function for pressure (q_test) and the divergence of the velocity field (u_answer) 
    # over the entire domain. For incompressible flow, this divergence should be zero.
    F1 = fe.inner(fe.div(u_answer), q_test) * dt * fe.dx

    return F1

def F2(u_answer, u_prev, p_answer, v_test, dt, rho1, n_v, mu1, gamma, alpha1, ds1, dx1):
    """
    Define the weak form of the momentum equation for the Navier-Stokes problem.

    Args:
    u_answer: dolfin.Function
        The current approximation of the velocity field in the mixed function space.
    u_prev: dolfin.Function
        The velocity field from the previous time step.
    p_answer: dolfin.Function
        The current approximation of the pressure field in the mixed function space.
    v_test: dolfin.TestFunction
        The test function for velocity in the mixed function space.
    dt: float
        The time step for the transient simulation.
    rho1: float
        The density of the fluid.
    n_v: dolfin.Constant or dolfin.Expression
        The normal vector used in the traction term.

    Returns:
    ufl.Form
        The weak form of the momentum equation suitable for FEniCS assembly.
    """
    
    F2 = (
        fe.inner((u_answer - u_prev) / dt, v_test) * fe.dx
        + fe.inner(fe.dot(u_answer, fe.grad(u_answer)), v_test) * fe.dx
        + (1/rho1) * fe.inner(sigma(u_answer, p_answer, mu1), epsilon(v_test)) * fe.dx
        # - (1/rho1) * fe.inner(Traction(T_answer, n_v, gamma), v_test) * ds1(1)
        # Uncomment the following lines if buoyancy force is needed
        # + fe.inner(gravity * alpha1 * (T_answer - T_ref), v_test[1]) * fe.dx  # Bouyancy y-component
        #Remeber alpha1 ?!
    )

    return F2

def solve_navier_stokes_heat_transfer(mesh, Bc, dt, upT, W, rho1, mu1, gamma, n_v, alpha1, Cp1, K1, absolute_tolerance, relative_tolerance, u_answer, u_prev, p_answer, v_test, q_test, ds1, dx1):
    """
    Solves the coupled Navier-Stokes and heat transfer problem using FEniCS.

    Args:
    mesh: fenics.Mesh
        The computational mesh.
    Bc: list
        List of Dirichlet boundary conditions.
    dt: float
        Time step for the transient simulation.
    upT: fenics.Function
        Function representing the current solution for velocity, pressure, and temperature.
    W: fenics.FunctionSpace
        Mixed function space for velocity, pressure, and temperature.
    rho1, mu1, gamma, alpha1, Cp1, K1: float
        Physical constants for the fluid.
    T_left, T_right, T_ref: float
        Temperatures for boundary conditions and reference temperature.

    Returns:
    upT: fenics.Function
        Updated function after solving the nonlinear variational problem.
    """

    # Define weak forms
    F1_form = F1(u_answer, q_test, dt)
    F2_form = F2(u_answer, u_prev, p_answer, v_test, dt, rho1, n_v, mu1, gamma, alpha1, ds1, dx1)

    # Define the combined weak form
    L = F1_form + F2_form 

    # Define the Jacobian
    J = derivative(L, upT)

    # Set up the nonlinear variational problem
    problem = NonlinearVariationalProblem(L, upT, Bc, J)

    # Set up the solver
    solver = NonlinearVariationalSolver(problem)

    # Set solver parameters
    prm = solver.parameters
    prm['newton_solver']['relative_tolerance'] = relative_tolerance
    prm['newton_solver']['absolute_tolerance'] = absolute_tolerance
    prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True



    return solver


#############################  END  ########################################

############################ Boundary Condition Section #################

def Define_Boundary_Condition(W, Domain, top_outflow_start, top_outflow_end, v_top, inflow_start, inflow_end, u_in  ) : 
    # Define the Domain boundaries based on the previous setup
    (X0, Y0), (X1, Y1) = Domain

    # Define boundary conditions for velocity, pressure, and temperature
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], X0)

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], X1)

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Y0)
        
    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Y1)

    class TopBoundaryNoslip(SubDomain):
        def inside(self, x, on_boundary):
            # Only the no-slip part of the top boundary, excluding the outflow section
            return on_boundary and near(x[1], Y1) and (x[0] < top_outflow_start or x[0] > top_outflow_end)
        
    class OutflowBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Only the outflow section of the top boundary
            return on_boundary and near(x[1], Y1) and (top_outflow_start <= x[0] <= top_outflow_end)

    class InflowBoundary(LeftBoundary):
        def __init__(self, inflow_start, inflow_end):
            self.inflow_start = inflow_start
            self.inflow_end = inflow_end
            super().__init__()

        def inside(self, x, on_boundary):
            return super().inside(x, on_boundary) and self.inflow_start <= x[1] <= self.inflow_end

    class NoSlipLeftBoundary(LeftBoundary):
        def __init__(self, inflow_start, inflow_end):
            self.inflow_start = inflow_start
            self.inflow_end = inflow_end
            super().__init__()

        def inside(self, x, on_boundary):
            return super().inside(x, on_boundary) and not (self.inflow_start <= x[1] <= self.inflow_end)



    # Instantiate boundary classes
    left_boundary = LeftBoundary()
    right_boundary = RightBoundary()
    bottom_boundary = BottomBoundary()
    no_slip_top_boundary = TopBoundaryNoslip()
    Out_flowBoundary = OutflowBoundary()
    inflow_boundary = InflowBoundary(inflow_start, inflow_end)
    no_slip_left_boundary = NoSlipLeftBoundary(inflow_start, inflow_end)
    top_boundary = TopBoundary()




    # Define Dirichlet boundary conditions
    bc_u_left = DirichletBC(W.sub(0), Constant((0, 0)), left_boundary)
    bc_u_right = DirichletBC(W.sub(0), Constant((0, 0)), right_boundary)
    bc_u_bottom = DirichletBC(W.sub(0), Constant((0, 0)), bottom_boundary)
    bc_u_top = DirichletBC(W.sub(0), Constant((0, 0 )), no_slip_top_boundary)
    bc_u_top_outflow = DirichletBC(W.sub(0), Constant((0,v_top)), Out_flowBoundary)
    bc_u_left_noslip = DirichletBC(W.sub(0), Constant((0, 0)), no_slip_left_boundary)
    bc_u_inflow = DirichletBC(W.sub(0), Constant((u_in , 0)) , inflow_boundary)






    # Point for setting pressure
    zero_pressure_point = fe.Point( X0, Y0  )
    bc_p_zero = DirichletBC(W.sub(1), Constant(0.0), lambda x, on_boundary: near(x[0], zero_pressure_point.x()) and near(x[1], zero_pressure_point.y()), method="pointwise")

    # Combine all boundary conditions

    bc_all = [bc_u_left, bc_u_right, bc_u_bottom, bc_u_top, bc_p_zero, bc_u_top_outflow, bc_u_left_noslip, bc_u_inflow ]

    # ******************************************
    # Create a MeshFunction for marking the subdomains


    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)

    # Mark the subdomains with the boundary objects
    bottom_boundary.mark(sub_domains, 2)  # Mark the bottom boundary with label 2
    top_boundary.mark(sub_domains, 1)  # Mark the top boundary with label 1

    # Define measures with the subdomain marking
    ds = Measure("ds", domain=mesh, subdomain_data=sub_domains)  # For boundary integration

    # Define an interior domain class to mark the interior of the domain
    class Interior(SubDomain):
        def inside(self, x, on_boundary):
            return not (top_boundary.inside(x, on_boundary) or bottom_boundary.inside(x, on_boundary))

    # Mark the interior domain
    domains2 = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains2.set_all(0)  # Initially mark all cells as 0
    interior_obj = Interior()
    interior_obj.mark(domains2, 1)  # Mark cells inside the interior domain as 1

    # Define the dx measure for the interior domain
    dx = Measure("dx", domain=mesh, subdomain_data=domains2)

    return ds, dx, bc_all

#############################  END  ################################


#################### Define Step 1 For Solving  ####################

W, v_test, q_test, upT, upT0, u_answer, p_answer, u_prev, p_prev = create_function_spaces(mesh)

n_v = Constant(( 0, 1 ) )

ds1, dx1, bc_all = Define_Boundary_Condition(W, Domain, top_outflow_start, top_outflow_end, v_top, inflow_start, inflow_end, u_in  )

solver = solve_navier_stokes_heat_transfer(
    mesh, bc_all, dt, upT, W, RHO1, MU1, GAMMA, n_v, ALPHA1, CP1, K1,  1E-6 , 1E-5,
      u_answer, u_prev, p_answer, v_test, q_test, ds1, dx1)



#############################  END  ###############################

#################### Define Initial Condition ####################

class InitialConditions(fe.UserExpression):
    """
    This class represents the initial conditions for the simulation.
    It initializes the velocity components, pressure, and temperature.
    """
    def eval(self, values, x):
        """
        Set the initial values for [velocity_x, velocity_y, pressure, temperature].
        Args:
            values: The array to be filled with the initial values.
            x: The coordinates where the initial values are evaluated.
        """
        values[0] = 0  # Initial x-component of velocity
        values[1] = 0  # Initial y-component of velocity
        values[2] = 0.0  # Initial pressure

    def value_shape(self):
        """
        Return the shape of the initial values array.
        This is a vector of length 4 for [velocity_x, velocity_y, pressure, temperature].
        """
        return (3,)

initial_v  = InitialConditions( degree = 2 ) 

upT.interpolate( initial_v )
upT0.interpolate( initial_v )

#############################  END  ################################

############################ File Section #########################


file = fe.XDMFFile("example.xdmf" ) # File Name To Save #


def write_simulation_data(Sol_Func, time, file, variable_names ):
    """
    Writes the simulation data to an XDMF file. Handles an arbitrary number of variables.

    Parameters:
    - Sol_Func : fenics.Function
        The combined function of variables (e.g., Phi, U, Theta).
    - time : float
        The simulation time or step to associate with the data.
    - file_path : str, optional
        The path to the XDMF file where data will be written.
    - variable_names : list of str, optional
        The names of the variables in the order they are combined in Sol_Func.
    """

    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()



T = 0

variable_names = [  "Vel", "Press" ]  # Adjust as needed


write_simulation_data( upT0, T, file , variable_names=variable_names )


#############################  END  ###############################


########################### Solving Loop  #########################



# Time-stepping loop
for it in tqdm(range(200000)):

    


    # Write data to file at certain intervals
    if it % 10 == 0:
        write_simulation_data(upT, T, file, variable_names)


    # Solve the system
    no_of_it, converged = solver.solve()

    # Update the previous solution
    upT0.vector()[:] = upT.vector()


    # Update time
    T = T + dt


    # Printing Informations Related to solutions behaviour

    u_max, u_min = compute_global_velocity_extremes(upT, W, comm)
    peclet_number, reynolds_number, CFL_condition = calculate_dimensionless_numbers(u_max, domain_length_x, K1, RHO1, MU1, grid_spacing_x)
    
    if rank == 0 and it% 100 ==0  :  # Only print for the root process

        print(" ├─ Iteration: " + str(it), flush=True)
        print(" Peclet Number is (Advective/Diffusive) Transport rate: " + str(peclet_number) , flush=True)
        print(" Reynolds Number is: " + str(reynolds_number), flush=True)
        print(" Based on CFL condition dt should be less than:  ", str(CFL_condition) , flush=True )

#############################  END  ###############################
