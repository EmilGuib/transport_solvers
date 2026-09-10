import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

#Functions
def angular_to_scalar_flux(angular_flux,quadrature_weights):
    num_cells_total = angular_flux.shape[1]
    scalar_flux = np.zeros(num_cells_total)
    for cell_index in range(num_cells_total):
        scalar_flux[cell_index] = np.sum(quadrature_weights*angular_flux[:,cell_index])
    return scalar_flux

def gauss_legendre_quadrature(num_angles):
    mu,quadrature_weights = roots_legendre(num_angles)
    return mu,quadrature_weights

def check_convergence(previous_scalar_flux,current_scalar_flux,relative_tolerance):
    if np.max(np.abs(previous_scalar_flux-current_scalar_flux)/(np.abs(current_scalar_flux)+1e-13)) < relative_tolerance:
        return 1
    else:
        return 0

def diamond_difference(incoming_angular_flux,sigma_t,mu,cell_width,angular_source):
    outgoing_angular_flux = ((2*abs(mu)-sigma_t*cell_width)*incoming_angular_flux+2*cell_width*angular_source)/(2*abs(mu)+sigma_t*cell_width)
    cell_angular_flux = 0.5*(outgoing_angular_flux+incoming_angular_flux)
    return outgoing_angular_flux,cell_angular_flux

def solve_source_iteration(region_widths,cells_per_region,sigma_s,sigma_t,external_source,num_angles,left_incoming_flux,right_incoming_flux):
    relative_tolerance = 1e-4
    max_iterations = 10000
    num_cells_total = np.sum(cells_per_region)
    sigma_s = np.repeat(sigma_s,cells_per_region)
    sigma_t = np.repeat(sigma_t,cells_per_region)
    cell_widths = np.repeat(region_widths/cells_per_region,cells_per_region)
    mu,quadrature_weights = gauss_legendre_quadrature(num_angles)
    external_source = np.repeat(external_source,cells_per_region)

    scalar_flux = np.zeros(num_cells_total)
    angular_flux = np.zeros((num_angles,num_cells_total))

    for iteration in range(1,max_iterations+1):
        negative_flux = False
        previous_scalar_flux = scalar_flux.copy()
        cell_angular_source = 0.5*(sigma_s*previous_scalar_flux+external_source)

        for angle_index in range(num_angles):
            if mu[angle_index] > 0:
                incoming_angular_flux = left_incoming_flux
                for cell_index in range(num_cells_total):
                    outgoing_angular_flux,cell_angular_flux = diamond_difference(incoming_angular_flux,sigma_t[cell_index],mu[angle_index],cell_widths[cell_index],cell_angular_source[cell_index])
                    if outgoing_angular_flux < 0:
                        negative_flux = True
                    angular_flux[angle_index,cell_index] = cell_angular_flux
                    incoming_angular_flux = outgoing_angular_flux

            if mu[angle_index] < 0:
                incoming_angular_flux = right_incoming_flux
                for cell_index in range(num_cells_total-1,-1,-1):
                    outgoing_angular_flux,cell_angular_flux = diamond_difference(incoming_angular_flux,sigma_t[cell_index],mu[angle_index],cell_widths[cell_index],cell_angular_source[cell_index])
                    if outgoing_angular_flux < 0:
                        negative_flux = True
                    angular_flux[angle_index,cell_index] = cell_angular_flux
                    incoming_angular_flux = outgoing_angular_flux

        scalar_flux = angular_to_scalar_flux(angular_flux,quadrature_weights)

        if check_convergence(previous_scalar_flux,scalar_flux,relative_tolerance):
            print("Number Iterations",iteration)
            return scalar_flux,negative_flux

    raise RuntimeError(f"Source iteration failed to converge after {max_iterations} iterations.")


#########################################################################################
#########################################################################################
#Numerical experiments outlined in document
#########################################################################################
#########################################################################################

#Pure Absorber
sigma_s = np.array([0])
sigma_t = np.array([0.942])
region_widths = np.array([1])
external_source = np.array([0])
cells_per_region = np.array([50])
num_angles = 100
left_incoming_flux = 1
right_incoming_flux = 0

scalar_flux,negative_flux = solve_source_iteration(region_widths,cells_per_region,sigma_s,sigma_t,external_source,num_angles,left_incoming_flux,right_incoming_flux)
print("Experiment 1: ",scalar_flux)
print()
#########################################################################################

#Internal Isotropic Source
sigma_s = np.array([0.5])
sigma_t = np.array([0.942])
region_widths = np.array([1])
external_source = np.array([1])
cells_per_region = np.array([50])
num_angles = 100
left_incoming_flux = 0
right_incoming_flux = 0

scalar_flux,negative_flux = solve_source_iteration(region_widths,cells_per_region,sigma_s,sigma_t,external_source,num_angles,left_incoming_flux,right_incoming_flux)
print("Experiment 2: ",scalar_flux)
print()
#########################################################################################

#Increased Scattering Ratio
sigma_t = np.array([1.0])
region_widths = np.array([10.0])
external_source = np.array([1.0])
cells_per_region = np.array([100])
num_angles = 16
left_incoming_flux = 0
right_incoming_flux = 0

scattering_ratios = np.array([0.00,0.20,0.50,0.70,0.85,0.90,0.95,0.98,0.99])

print("Experiment #3")
for scattering_ratio in scattering_ratios:
    sigma_s = scattering_ratio*sigma_t
    scalar_flux,negative_flux = solve_source_iteration(region_widths,cells_per_region,sigma_s,sigma_t,external_source,num_angles,left_incoming_flux,right_incoming_flux)
#########################################################################################

#Mesh Refinement
sigma_s = np.array([0.5])
sigma_t = np.array([0.942])
region_widths = np.array([1])
external_source = np.array([0])
num_angles = 100
left_incoming_flux = 1
right_incoming_flux = 1

print("Experiment #4")
for cells in [1,5,8,10,50,100,500]:
    cells_per_region = np.array([cells])
    scalar_flux,negative_flux = solve_source_iteration(region_widths,cells_per_region,sigma_s,sigma_t,external_source,num_angles,left_incoming_flux,right_incoming_flux)
    print(f"Negative Flux?: {negative_flux}")
    cell_centers = (np.arange(cells)+0.5)*region_widths[0]/cells
    print(f"{cells:4d} cells: negative flux = {negative_flux}")
    plt.plot(cell_centers,scalar_flux,label=f"{cells} cells")

plt.xlabel("x [cm]")
plt.ylabel("Scalar Flux")
plt.legend()
plt.show()
#########################################################################################

#Multi-Layer Slab
#Reed Problem
sigma_s = np.array([0.0,0.0,0.0,0.9,0.9])
sigma_t = np.array([50.0,5.0,0.0,1.0,1.0])
region_widths = np.array([2.0,1.0,2.0,1.0,2.0])
external_source = np.array([50.0,0.0,0.0,1.0,0.0])
cells_per_region = np.array([50,50,50,50,50])
num_angles = 8
left_incoming_flux = 1
right_incoming_flux = 0

scalar_flux,negative_flux = solve_source_iteration(region_widths,cells_per_region,sigma_s,sigma_t,external_source,num_angles,left_incoming_flux,right_incoming_flux)

#Position of each cell center
cell_widths = np.repeat(region_widths/cells_per_region,cells_per_region)
x_edges = np.concatenate(([0.0],np.cumsum(cell_widths)))
cell_centers = 0.5*(x_edges[:-1]+x_edges[1:])

#Plot
plt.figure(figsize=(9,5))
plt.plot(cell_centers,scalar_flux,linewidth=2,label="Scalar Flux")

#Material interfaces
interfaces = np.cumsum(region_widths)[:-1]
for x_interface in interfaces:
    plt.axvline(x_interface,linestyle="--",linewidth=1)

plt.xlabel("Position, x [cm]")
plt.ylabel("Scalar Flux")
plt.title("Experiment 5: Multi-Layer Slab")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#########################################################################################