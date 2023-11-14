
""" MECA.py     => This script the Modified Equivalent Current Approximation method (MECA) [1]. An application of this algorithm can be found in [2].

Author: Rasul Choupanzadeh 
Date: 08/13/2023

Acknowledgement 1: This project is completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol,
                   in the Applied Computational Electromagnetics and Signal/Power Integrity (ACEM-SPI) Lab while working toward the
                   Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA.

Acknowledgement 2: This algorithm is written based the expressions of [1]. However, some of the expressions in [1] were
                   identified incorrect and the corrections are presented in [2]. 


[1]   J. G. Meana, J. A. Martinez-Lorenzo, F. Las-Heras, and C. Rappaport, “Wave scattering by dielectric and lossy materials using
      the modified equivalent current approximation (MECA),” IEEE Trans. Antennas Propag., vol. 58, no. 11, pp. 3757–3761, Nov. 2010.

[2]  R. Choupanzadeh and A. Zadehgol, "A Deep Neural Network Modeling Methodology for Efficient EMC Assessment of Shielding Enclosures
     Using MECA-Generated RCS Training Data," IEEE Transactions on Electromagnetic Compatibility, DOI: 10.1109/TEMC.2023.3316916.
        
[3]  A. A. Kishk, Electromagnetic Waves Propagation in Complex Matter, 1st ed. Rijeka, Croatia: InTech, 2011.     

[4]  C. Geuzaine and J.-F. Remacle. "Gmsh: a three-dimensional finite element mesh generator with built-in pre- and
     post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), pp. 1309-1331, 2009.
     
[5]  C. A. Balanis, Advanced Engineering Electromagnetics, 2nd ed. Hoboken, NJ, USA: Wiley, 2012.

[6]  C. A. Balanis, Antenna Theory: Analysis and Design, 4th ed. Hoboken, NJ, USA: Wiley, 2016.

[7]  Gmsh website: https://gmsh.info

[8]  CST-Computer Simulation Technology AG, “CST studio suite 2022,” 2022.
     Accessed: Apr. 27, 2023. [Online]. Available: https://www.3ds.com/products-services/simulia/products/cst-studio-suite/

"""

## Input: Medium 1 specifications (epsr_1, mur_1, sigma_1), Medium 2 specs (epsr_2, mur_2, sigma_2), Incident Wave specifications (teta_inc_deg, E0_TE_inc, E0_TM_inc),   
##        Frequency: f, Structure specifications (structure, x_dim, y_dim, h, l_app, w_app, max_cell_size, aperture), CST results for validation of Box example                                     Output: RCS



import numpy as np
import matplotlib.pyplot as plt
import math



def car_to_spher(arr):
    x=arr[0]; y=arr[1]; z=arr[2]
    r = np.sqrt( x ** 2 + y ** 2 + z ** 2 )
    theta = np.arccos(z/r)
    phi = np.sign(y) * np.arccos(x/np.sqrt(x**2 + y**2))
    res = np.array([r,theta,phi])
    return res

def cartesian_to_spherical(arr):
    x=arr[0]; y=arr[1]; z=arr[2]
    r = np.sqrt( x ** 2 + y ** 2 + z ** 2 )
    theta = math.atan2(math.sqrt(x ** 2 + y ** 2), z)
    phi = math.atan2(y, x) if x >= 0 else math.atan2(y, x) + math.pi
    if phi <0:
        phi = phi + 2*np.pi   
    arr = np.array([r,theta,phi])
    return arr

def spherical_to_cartesian(arr):
    r_mag = arr[0]
    theta = arr[1]
    phi = arr[2]
    x = r_mag * np.cos(phi) * np.sin(theta)
    y = r_mag * np.sin(phi) * np.sin(theta)
    z = r_mag * np.cos(theta)                #np.round(np.cos(theta),decimals=10)
    return x, y, z


def disp_vector(P1,P2):
    [r1,theta1,phi1]=P1
    [r2,theta2,phi2]=P2
    R21_x = -r1*np.cos(phi1)*np.sin(theta1) + r2*np.cos(phi2)*np.sin(theta2) 
    R21_y = -r1*np.sin(phi1)*np.sin(theta1) + r2*np.sin(phi2)*np.sin(theta2)
    R21_z = -r1*np.cos(theta1) + r2*np.cos(theta2)
    R21 = np.array([R21_x,R21_y,R21_z])
    return R21


def jacob(arr_cart, teta, phi):
    A = np.array([[np.sin(teta)*np.cos(phi), np.sin(teta)*np.sin(phi), np.cos(teta)],
                  [np.cos(teta)*np.cos(phi), np.cos(teta)*np.sin(phi), -np.sin(teta)],
                  [-np.sin(phi),np.cos(phi),0]])
    arr_sph = np.matmul(A, arr_cart)  
    return arr_sph

##-----------------------------------------Inputs----------------------------------------------------
eps0 = 1e-9/(36*np.pi)     # exactly => eps0 = 8.854*1e-12
mu0 = 4*np.pi*1e-7

# Medium 1 (Air)
epsr_1 = 1                  
mur_1 = 1
sigma_1 = 0                 

# Medium 2 (PEC sigma_2= infinite, eps= infinite )
sigma_2 = 1e100   
epsr_2 = 1e100                
mur_2 = 1
       
# Incidence angle
teta_inc_deg = 30                        # in Degree
teta_inc = np.deg2rad(teta_inc_deg)      # in Radian
#phi_inc = np.deg2rad(270)               # it is considered 270 deg. for RCS calculations in phi_scattering = 90 deg.


# Plate coordination
n_unit = np.array([0,0,1])
u_unit = np.array([0,1,0])


# Incident wave (uniform plane wave)
k_inc_unit = np.array([0, np.sin(teta_inc), -np.cos(teta_inc)])                             # Oblique incident to x-y plane, k_inc_unit = np.sin(teta_inc)* u_unit - np.cos(teta_inc)*n_unit 
E0_TE_inc = 0
E0_TM_inc = 1
E0 = np.linalg.norm(E0_TM_inc+E0_TE_inc)

# Frequency
f = 3.6e9
lambda1 = 3e8/f

# Decomposition to TE and TM 
e_te_unit = np.cross(k_inc_unit,u_unit)/np.linalg.norm(np.cross(k_inc_unit,u_unit))         # np.sqrt(x.dot(x)) is much faster than np.linalg.norm() to calculate the magnitude of a vector   
e_tm_unit = np.cross(e_te_unit, k_inc_unit)


##---------------------------------Parameter Calculation-----------------------------------------------
eps_1 = epsr_1*eps0
eps_2 = epsr_2*eps0
eps = np.array([eps_1, eps_2])
mu_1 = mur_1*mu0
mu_2 = mur_2*mu0
mu = np.array([mu_1, mu_2])
sigma = np.array([sigma_1, sigma_2])

beta = np.zeros(shape=np.shape(eps), dtype='complex64')
etta = np.zeros(shape=np.shape(eps), dtype='complex64')     
for i in range(len(eps)):
    cons = sigma[i]/(2*np.pi*f*eps[i])      
    beta[i] = 2*np.pi*f*np.sqrt(mu[i]*eps[i]) * np.sqrt(0.5*(np.sqrt(1+cons**2)+1))
    etta[i] = np.sqrt(1j*2*np.pi*f*mu[i]/(sigma[i]+1j*2*np.pi*f*eps[i]))

etta_1 = etta[0]
etta_2 = etta[1]
beta_1 = beta[0]
beta_2 = beta[1]

k1 = beta_1
k2_t = (2*np.pi*f)**2 * mu_2 * (eps_2 - 1j*sigma_2/(2*np.pi*f) )

Re_k_t_w = np.sqrt( 0.5*(np.real(k2_t) -k1**2 * np.sin(teta_inc)**2 + np.sqrt( (np.real(k2_t)-k1**2 * np.sin(teta_inc)**2)**2 + np.imag(k2_t) ) ) )
Im_k_t_w = np.imag(k2_t)/(2*Re_k_t_w)

k_t_w = Re_k_t_w + 1j*Im_k_t_w    

# Eq. (22) & (23) in [1]
R_TE = (mu_2*k1*np.cos(teta_inc) - mu_1*k_t_w) / (mu_2*k1*np.cos(teta_inc) + mu_1*k_t_w)
R_TM = - ( (eps_2-1j*sigma_2/(2*np.pi*f)) *k1*np.cos(teta_inc) - eps_1*k_t_w ) / ( (eps_2-1j*sigma_2/(2*np.pi*f))*k1*np.cos(teta_inc) + eps_1*k_t_w )

def chomp(R):
    """
        Remove any noisy close to zero imaginary values
    """
    check = np.isclose(R.imag/R.real, 0, atol=1e-10)      # Removes noisy imaginary values < 10^-10
    if check == True:
        R = R.real
    return R

R_TE = np.round(chomp(R_TE),10)
R_TM = np.round(chomp(R_TM),10)


##--------------------------------------------Define Gmsh inputs to generate mesh [4]--------------------------------------------------------
# Dimensions of structure (a 5lambda*5lambda*5lambda Cube)
l = 5*lambda1

x_dim = 1*l             # x-direction plate length
y_dim = 1*l             # y-direction plate length
h = 1*l                   # heigh of enclosure in z-direction

# Aperture size
l_app = x_dim / 2       # x-direction aperture length
w_app = y_dim / 5       # y-direction aperture length

aperture = True        # True = Enable aperture,     Flase = Disable aperture 

# Mesh cell size in Gmesh
max_cell_size = lambda1 

# Run Gmes
structure = "gmsh_box.py"
#structure = "gmsh_parallel_plates.py"
exec(open(structure).read())

num_facets =  num_Elements       # or num_facets = mesh_data.shape[0]


##---------------------------------Calculate scattered field and RCS-----------------------------------------------          
# Scattering Angle
teta_s_deg = np.arange(0,181,1)   
phi_s = np.pi/2

# Observation point
farfield = 4*l**2/lambda1
r_ob_mag = 10*farfield

# RCS Calculation
RCS = np.zeros(shape=(len(teta_s_deg)))
RCS_cart = np.zeros(shape=(len(teta_s_deg)))
Es_phi = np.zeros(shape=(len(teta_s_deg)), dtype='complex64')
I = np.zeros(shape=(len(teta_s_deg)),dtype='complex64')

for teta_index in teta_s_deg:
    print(teta_index)
    teta_s0 = np.deg2rad(teta_index)
    r_k = np.array([r_ob_mag*np.sin(teta_s0)*np.round(np.cos(phi_s), decimals = 10), r_ob_mag*np.sin(teta_s0)*np.sin(phi_s) ,r_ob_mag*np.round(np.cos(teta_s0), decimals=10)])        #  observation point
    r_k_unit = np.array([np.sin(teta_s0)*np.round(np.cos(phi_s),decimals=10), np.sin(teta_s0)*np.sin(phi_s), np.round(np.cos(teta_s0),decimals=10)]) 

    p_inc_unit = k_inc_unit 
    Es = np.zeros(3, dtype='complex64')
    Hs = np.zeros(3, dtype='complex64')
    I_tet = 0
    
    for i in range(num_facets):
        P1 = mesh_data[i,0].copy()
        P2 = mesh_data[i,1].copy()
        P3 = mesh_data[i,2].copy()
        barycenter = (P1+P2+P3)/3                               # barycenter (centroid) of a triangle = ( (x1+x2+x3)/3 , (y1+y2+y3)/3 )  or barycenter= bary_data[i]
        
        V12 = P2 - P1
        V13 = P3 - P1      
        
        #Area = 0.5 * np.abs(P1[0]*(P2[1]-P3[1]) + P2[0]*(P3[1]-P1[1]) + P3[0]*(P1[1]-P2[1]))  # Area of triangle with given coordinates => A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        Area = 0.5 * np.linalg.norm(np.cross(V12,V13))          # This is general formula (3D) for area of a facet, previous formula works only for 2D structures
        A = Area
        
        n_i_unit = np.cross(V12,V13) / np.linalg.norm(np.cross(V12,V13))
        n_i_unit = np.sign(np.dot(bary_data[i],n_i_unit)) * n_i_unit
        n_i_unit = np.round(n_i_unit, decimals = 10)
        #if n_i_unit[2] != 0:
            #n_i_unit = np.array([0, 0, 1])
        
        r_i = barycenter
        
        # Calculate modified equivalent currents through eq. (6) and (7) in [1]
        M_i0 = ( E0_TE_inc*(1+R_TE)* np.cross(e_te_unit, n_i_unit)  +  E0_TM_inc*np.cos(teta_inc)*(1+R_TM)* e_te_unit ) * np.exp(-1j*k1*(r_i[1]*np.sin(teta_inc)-r_i[2]*np.cos(teta_inc)))  
        J_i0 = (1/etta_1)* ( E0_TE_inc*np.cos(teta_inc)*(1-R_TE)* e_te_unit  +  E0_TM_inc*(1-R_TM)* np.cross(n_i_unit, e_te_unit) ) * np.exp(-1j*k1*(r_i[1]*np.sin(teta_inc)-r_i[2]*np.cos(teta_inc))) 
        
        J_i0 = np.real_if_close(J_i0, tol = 1e-10)              # removes 0j in PEC case for faster calculations
        M_i0 = np.real_if_close(M_i0, tol = 1e-10)
        
        # Calculate I_r based on the eq.(32) and corresponding table of eq.(31) in [3]
        r_i_unit = r_i/np.linalg.norm(r_i)

        r_ik = r_k - r_i
        r_ik_unit = r_ik/np.linalg.norm(r_ik)  
        r_ik_unit = np.round(r_ik_unit, decimals = 8) 

        a = (2*np.pi /lambda1)*np.dot(V12 , r_ik_unit - p_inc_unit)
        b = (2*np.pi /lambda1)*np.dot(V13 , r_ik_unit - p_inc_unit)
        
        ## a and b MUST be rounded. For example a=9.140795493725524e-17 and b=-5.270247302722068e-17 must result I_r = A but results I_r=(3.550913230632813e+16-0.458132692707869j)*A!!
        a = np.round(a, decimals = 10)
        b = np.round(b, decimals = 10)
        
        if a==0 and b ==0:
            I_r = A
        elif a==0 and b !=0:
            I_r = 2*A* np.exp(-1j*b/3) * ( (1 + 1j*b - np.exp(1j*b)) / (b**2) )
        elif a!=0 and b ==0:
            I_r = 2*A* np.exp(-1j*a/3) * ( (1 + 1j*a - np.exp(1j*a)) / (a**2) )   
        elif a!=0 and b==a:
            I_r = 2*A* np.exp(-1j*2*a/3) * ( (np.exp(1j*a)*(1-1j*a) - 1) / (a**2) )  
        else:
            I_r = 2*A* np.exp(-1j*(a+b)/3) * ( ( a*np.exp(1j*b) - b*np.exp(1j*a) + b-a ) / ((a-b)*a*b) )


        # Eq.(4) in [2]
        Ea_ik =  np.exp(1j*k1* np.dot(r_k_unit,r_i)) * I_r * np.cross(r_k_unit, M_i0)          
        Ha_ik =  np.exp(1j*k1* np.dot(r_k_unit,r_i)) * I_r * np.cross(r_k_unit, J_i0)         

        # Eq.(3) in [2]
        Es = Es + (1j/(2*lambda1)) * (np.exp(-1j*k1*np.linalg.norm(r_k))/np.linalg.norm(r_k)) * (Ea_ik - etta_1 * np.cross(Ha_ik,r_k_unit))
        Hs = Hs + (-1j/(2*lambda1)) * (np.exp(-1j*k1*np.linalg.norm(r_k))/np.linalg.norm(r_k)) * (Ha_ik - (1/etta_1)* np.cross(r_k_unit,Ea_ik))  
    
    # Eq.(24) in [1]
    RCS_cart[teta_index]  = 4*np.pi* r_ob_mag**2 * np.dot(np.abs(Es),np.abs(Es))/np.dot(E0,E0)

# Convert RCS_cart (RCS calculated in cartesian corrdinates) from scalar to dB 
RCS_db_cart = 10*np.log10(RCS_cart)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(teta_s_deg,RCS_db_cart,label='MECA ',color='red', linestyle='dashed', linewidth=2)


##---------------------------------------------------------Import CST results [8]------------------------------------------------------
if structure == "gmsh_box.py":
    file_name = 'CST_obliqueCube.txt'      
    CST = open(file_name).readlines()  
    CST = [line.split() for line in CST]                # removes the empty columns (space between numbers)
    CST = [ele for ele in CST if ele != []]             # removes the empty rows (empty lines)
    
    teta_CST = np.zeros(shape=(len(CST)))
    RCS_CST = np.zeros(shape=(len(CST)))
    
    for i in range(2, len(CST)):
        teta_CST[i] = float(CST[i][0])
        RCS_CST[i] = float(CST[i][2])
    ax.plot(teta_CST,RCS_CST,label='CST',color='blue',linestyle='solid',  linewidth=2)  # linestyle=(0,(7, 7))

    ##---------------------------plot configurations----------------------------
    plt.xlim(teta_s_deg[0], teta_s_deg[-1])
    plt.ylim(-50,40)
    plt.xlabel('$\u03B8_s$ (\u00B0)', fontsize=16)
    plt.ylabel('RCS (dBm$^2$)', fontsize=16)
    plt.legend(loc='lower left',fontsize=14, frameon=True, edgecolor='black', shadow=False) #loc='lower right', 
    plt.grid(linestyle = '--')
    plt.tick_params(axis='both', labelsize=12)
    
    plt.savefig('oblique_cube_'+str(teta_inc_deg) + 'degree.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()



##------------------------------------Balanis EM (eq 11.44, 11.38a, 11.38b in [5]) + Array theory (sec 6.2 in [6])-------------------------------------
## This is for TM mode in MECA (equivalent to TE mode in Balanis) 
if structure == "gmsh_parallel_plates.py":
    # RCS of a plate in x-z plane (TM of MECA, or TE of Balanis)
    a =  x_dim
    b =  y_dim
    k=k1.real
    RCS_Bal = np.zeros(shape=(len(teta_s_deg)))
    E_Bal_2 = np.zeros(shape=(len(teta_s_deg)),dtype='complex64')
    tet_i = 0
    for teta_index in teta_s_deg:
        teta_s0 = np.deg2rad(teta_index)
        X = (k*a/2)*(np.sin(teta_s0)*np.cos(phi_s))
        Y = (k*b/2)*(np.sin(teta_s0)*np.sin(phi_s) - np.sin(teta_inc)) 
        RCS_Bal_teta_s0 = 4*np.pi* (a*b/lambda1)**2 * ( np.round(np.cos(teta_s0), decimals = 10)**2 *np.sin(phi_s)**2 + np.round(np.cos(phi_s), decimals = 10) **2 ) * np.sinc(X/np.pi)**2 * np.sinc(Y/np.pi)**2 * 4 * np.sin((np.pi/lambda1)*h*np.cos(teta_s0) + (np.pi/lambda1)*h*np.round(np.cos(teta_inc), decimals = 10)) ** 2
        RCS_Bal[tet_i] = RCS_Bal_teta_s0
        tet_i = tet_i + 1
    
    RCS_Bal_db = 10*np.log10(RCS_Bal)
    ax.plot(teta_s_deg,RCS_Bal_db,label='Analyt. Form.',color='black',linestyle='solid',  linewidth=1)  # linestyle=(0,(7, 7))  #, marker='.', markersize=1
    plt.xlim(teta_s_deg[0], teta_s_deg[-1])
    plt.ylim(-130, 30)
    plt.title('Bistatic RCS of two parallel plates ($\u03B8_{inc}$ = '+str(teta_inc_deg) +'\u00B0)', fontsize=16)
    plt.xlabel('$\u03B8_s$ (\u00B0)', fontsize=16)
    plt.ylabel('RCS (dBm$^2$)', fontsize=16)
    plt.legend(loc='lower left',fontsize=14, frameon=True, edgecolor='black', shadow=False) 
    plt.grid(linestyle = '--') 
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('two_parallel_plates_'+str(teta_inc_deg) + 'degree.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()