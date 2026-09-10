import numpy as np
import matplotlib.pyplot as plt
from math import pi as pi


def Angular_Flux_To_Scalar(angular_flux, dTheta):
    num_cells_total_x = angular_flux.shape[1]
    num_cells_total_y = angular_flux.shape[2]
    scalar_flux = np.zeros((num_cells_total_x, num_cells_total_y))
    for i in range(num_cells_total_x):
        for j in range(num_cells_total_y):
            scalar_flux[i, j] = np.sum(dTheta * angular_flux[:, i, j])

    return scalar_flux

def Mu_and_dTheta(N):
    d_theta = 2*pi / N
    theta_m = 2*np.pi*(np.arange(N) + 0.5)/N
    mu_x = np.cos(theta_m)
    mu_y = np.sin(theta_m)
    return mu_x, mu_y, d_theta

def Check_Convergence(previous_scalar_flux,current_scalar_flux,tolerance):
    if np.max(np.abs(previous_scalar_flux-current_scalar_flux)/(np.abs(current_scalar_flux)+10e-14)) < tolerance:
        return True
    else:
        return False
    
def Diamond_Difference(angular_flux_x_in,angular_flux_y_in,sigma_t,q,mu_x,mu_y,dx,dy):
    mean_angular_flux = (q*dx*dy+2*(abs(mu_x)*dy*angular_flux_x_in+abs(mu_y)*dx*angular_flux_y_in))
    mean_angular_flux = mean_angular_flux / (2*(abs(mu_x)*dy+abs(mu_y)*dx)+sigma_t*dx*dy)
    angular_flux_x_out = 2*mean_angular_flux-angular_flux_x_in
    angular_flux_y_out = 2*mean_angular_flux-angular_flux_y_in
    return mean_angular_flux, angular_flux_x_out,angular_flux_y_out

def Expand_Regions(values, cells_x, cells_y):
    return np.repeat(
        np.repeat(values, cells_y, axis=0),
        cells_x,
        axis=1
    ).T


def Solve_Source_Iteration(region_widths,cells_per_region,sigma_s,sigma_t,q_ext, num_angles, 
                           left_incoming_fluxes, right_incoming_fluxes, bottom_incoming_fluxes, top_incoming_fluxes):
    
    
    tolerance = 1e-4
    max_iterations = int(1e6)
    
    sigma_s = Expand_Regions(sigma_s, cells_per_region[0], cells_per_region[1])
    sigma_t = Expand_Regions(sigma_t, cells_per_region[0], cells_per_region[1])
    q_ext = Expand_Regions(q_ext, cells_per_region[0], cells_per_region[1])
    
    num_x_cells = np.sum(cells_per_region[0])
    num_y_cells = np.sum(cells_per_region[1])
    dx = np.repeat(region_widths[0] / cells_per_region[0],cells_per_region[0])
    dy = np.repeat(region_widths[1] / cells_per_region[1],cells_per_region[1])
    mu_x, mu_y, d_theta = Mu_and_dTheta(num_angles)
    
    angular_flux = np.zeros((num_angles,np.sum(cells_per_region[0]),np.sum(cells_per_region[1])))
    scalar_flux = np.zeros((np.sum(cells_per_region[0]),np.sum(cells_per_region[1])))
    scalar_flux_previous = np.zeros((np.sum(cells_per_region[0]),np.sum(cells_per_region[1])))
                                 
    for current_iteration in range(0,max_iterations):
        scalar_flux_previous = scalar_flux.copy()
        q = (1/(2*pi)) * (sigma_s * scalar_flux_previous+q_ext)
        
        for angle_m in range(0,num_angles):
#########################################################################################
            if mu_x[angle_m] > 0:
                #Sweeping left to right, going upwards
                if mu_y[angle_m]> 0:
                    angular_flux_y_in = np.full(num_x_cells,bottom_incoming_fluxes,dtype=float)
                    for cell_j in range(num_y_cells):
                        angular_flux_x_in = left_incoming_fluxes
                        for cell_i in range(num_x_cells):
                                mean_angular_flux, angular_flux_x_out,angular_flux_y_out= Diamond_Difference(angular_flux_x_in,angular_flux_y_in[cell_i],sigma_t[cell_i,cell_j],q[cell_i,cell_j],mu_x[angle_m],mu_y[angle_m],dx[cell_i],dy[cell_j])
                                angular_flux[angle_m,cell_i,cell_j] = mean_angular_flux
                                angular_flux_x_in = angular_flux_x_out
                                angular_flux_y_in[cell_i] = angular_flux_y_out

                #Sweeping left to right, going downwards
                if mu_y[angle_m]< 0:
                    angular_flux_y_in = np.full(num_x_cells,top_incoming_fluxes,dtype=float)
                    for cell_j in range(num_y_cells-1,-1,-1):
                        angular_flux_x_in = left_incoming_fluxes
                        for cell_i in range(num_x_cells):
                                mean_angular_flux, angular_flux_x_out,angular_flux_y_out= Diamond_Difference(angular_flux_x_in,angular_flux_y_in[cell_i],sigma_t[cell_i,cell_j],q[cell_i,cell_j],mu_x[angle_m],mu_y[angle_m],dx[cell_i],dy[cell_j])
                                angular_flux[angle_m,cell_i,cell_j] = mean_angular_flux
                                angular_flux_x_in = angular_flux_x_out
                                angular_flux_y_in[cell_i] = angular_flux_y_out                                        
                                        
######################################################################################### 
            
            if mu_x[angle_m] < 0:
                #Sweeping right to left, going upwards
                if mu_y[angle_m]> 0:
                    angular_flux_y_in = np.full(num_x_cells,bottom_incoming_fluxes,dtype=float)
                    for cell_j in range(num_y_cells):
                        angular_flux_x_in = right_incoming_fluxes
                        for cell_i in range(num_x_cells-1,-1,-1):
                                mean_angular_flux, angular_flux_x_out,angular_flux_y_out= Diamond_Difference(angular_flux_x_in,angular_flux_y_in[cell_i],sigma_t[cell_i,cell_j],q[cell_i,cell_j],mu_x[angle_m],mu_y[angle_m],dx[cell_i],dy[cell_j])
                                angular_flux[angle_m,cell_i,cell_j] = mean_angular_flux
                                angular_flux_x_in = angular_flux_x_out
                                angular_flux_y_in[cell_i] = angular_flux_y_out
                #Sweeping right to left, going downwards
                if mu_y[angle_m]< 0: 
                    angular_flux_y_in = np.full(num_x_cells,top_incoming_fluxes,dtype=float)
                    for cell_j in range(num_y_cells-1,-1,-1):
                        angular_flux_x_in = right_incoming_fluxes
                        for cell_i in range(num_x_cells-1,-1,-1):
                                mean_angular_flux, angular_flux_x_out,angular_flux_y_out= Diamond_Difference(angular_flux_x_in,angular_flux_y_in[cell_i],sigma_t[cell_i,cell_j],q[cell_i,cell_j],mu_x[angle_m],mu_y[angle_m],dx[cell_i],dy[cell_j])
                                angular_flux[angle_m,cell_i,cell_j] = mean_angular_flux
                                angular_flux_x_in = angular_flux_x_out
                                angular_flux_y_in[cell_i] = angular_flux_y_out 
    
        scalar_flux = Angular_Flux_To_Scalar(angular_flux,d_theta)
        if Check_Convergence(scalar_flux_previous,scalar_flux,tolerance):
            return scalar_flux
        

        
if __name__ == "__main__":
    #########################################################################################
    #########################################################################################
    #Numerical experiments outlined in document
    #########################################################################################
    #########################################################################################
    sigma_s = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],[0.3, 0.3, 0.3, 0.3, 0.3]])
    sigma_t = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0, 1.0]])

    region_widths = (np.array([2.0, 1.0, 2.0, 1.0, 2.0]), np.array([2.0, 2.0]))
    cells_per_region = (np.array([50, 50, 50, 50, 50]),np.array([50, 50]))

    num_angles = 30
    q_ext = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]])
    left_incoming_fluxes = 1
    right_incoming_fluxes = 1
    bottom_incoming_fluxes = 1
    top_incoming_fluxes = 1

    scalar_flux = Solve_Source_Iteration(region_widths,cells_per_region,sigma_s,sigma_t,q_ext, num_angles, 
                               left_incoming_fluxes, right_incoming_fluxes, bottom_incoming_fluxes, top_incoming_fluxes)

    # ============================================
    # Plot scalar flux as a 2-D color map
    # ============================================

    num_x_cells = np.sum(cells_per_region[0])
    num_y_cells = np.sum(cells_per_region[1])

    dx = np.repeat(region_widths[0] / cells_per_region[0], cells_per_region[0])
    dy = np.repeat(region_widths[1] / cells_per_region[1], cells_per_region[1])

    x_edges = np.concatenate(([0.0], np.cumsum(dx)))
    y_edges = np.concatenate(([0.0], np.cumsum(dy)))

    X, Y = np.meshgrid(x_edges, y_edges, indexing='ij')

    plt.figure(figsize=(8, 6))
    pcm = plt.pcolormesh(X, Y, scalar_flux, shading='auto')
    plt.colorbar(pcm, label='Scalar Flux')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2-D Scalar Flux')
    plt.tight_layout()
    plt.show()