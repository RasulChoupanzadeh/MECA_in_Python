# Modified Equivalent Current Approximation (MECA)
### Author: Rasul Choupanzadeh
### Date: 11/13/2023

# Acknowledgement
This project is completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, in the Applied Computational Electromagnetics and Signal/Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. 

# Overview
- This algorithm has two main parts, as follows:
    1. Modified Equivalent Current Approximation (MECA) algorithm [1,2]
    2. Gmsh algorithm [3,4]

- The MECA algorithm implements an efficient numerical method to calculate the Radar Cross Section (RCS) of a structure in the farfield region. It calculates:
    * The reflection coefficients 
    * The modified current approximations
    * The scattered fields 
    * The RCS

- The Gmsh algorithm implements a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities. It calculates:
    * The mesh element coordinates
    * The barycenter (centroid) of mesh elements

- Finally, we provide validation data for the computed calculated RCS results by comparing against solution from CST Studio Suite "oblique_cube_30degree.pdf" [5], and solution from analytical formula in "two_parallel_plates_30degree.pdf" [6,7,8].
- **Note:** In [6], we apply MECA to an example and evaluate its efficiency compared to CST.

# Licensing
Licensed under GNU GPL v.3.

# Files:

## Primary Program Files:
- **MECA.py:** Implements the MECA algorithm for RCS calculation of the specified structure.
    * Inputs: 
		- Medium specifications (epsr_1,mur_1,sigma_1,epsr_2,mur_2,sigma_2)
		- Incident Wave specifications (teta_inc_deg,E0_TE_inc,E0_TM_inc)
		- Structure specifications (structure, x_dim, y_dim, h, l_app, w_app, max_cell_size, aperture)
		- frequency (f)
		- CST_obliqueCube.txt for validation of cube example
    * Outputs: 
		- RCS comparison of MECA and CST Studio Suite for cube example
		- RCS comparison of MECA and analytical expression for parallel plate example  
- **gmsh_box.py:** Generates meshes through Gmsh algorithm for the example of cube structure 
    * Inputs: x_dim, y_dim, h, l_app, w_app, max_cell_size, aperture 
    * Outputs: num_facets, mesh_data, bary_data
- **gmsh_parallel_plates.py:** Generates meshes through Gmsh algorithm for the example of cube structure 
    * Inputs: x_dim, y_dim, h, l_app, w_app, max_cell_size, aperture
    * Outputs: num_facets, mesh_data, bary_data
- **CST_obliqueCube.txt:** Simulation results of CST Studio Suite for a PEC cube under oblique incidence.
- **oblique_cube_30degree.pdf:** Compares the results of MECA with CST Studio Suite simulation results for a PEC cube under oblique incidence.
- **two_parallel_plates_30degree.pdf:** Compares the results of MECA with analytical formula (eq.(6) in [6]) for two PEC parallel plates under oblique incidence. 


# Run instructions
Two types of structures (cube and parallel plates) are predefined in Gmsh algorithm. To run the predefined structurs, you only need to run the "MECA.py" file for your desired incidence field, mediums, and frequency. To simulate other structures, you need to define them in Gmsh algorithm and simulate the MECA by replacing "structure= gmsh_newstructure.py" in Gmsh section of "MECA.py".


## Inputs:
- **epsr_1:** Relative permittivity of medium 1.
- **mur_1:** Relative permeability  of medium 1.
- **sigma_1:** Conductivity of medium 1.
- **epsr_2:** Relative permittivity of medium 2.
- **mur_2:** Relative permeability  of medium 2.
- **sigma_2:** Conductivity of medium 2.
- **teta_inc_deg:** Incidence angle in degrees.
- **E0_TE_inc:** TE components of incidence wave.
- **E0_TM_inc:** TM components of incidence wave.
- **structure:** Two predefined structure types (gmsh_box.py, gmsh_parallel_plates.py) in Gmsh algorithm. 
- **x_dim:** Dimension of structure in x-direction (width). 
- **y_dim:** Dimension of structure in x-direction (length). 
- **h:** Dimension of structure in z-direction (height). 
- **aperture:** Activates/deactivates the presence of the aperture.
- **l_app:** Length of apperture. 
- **w_app:** Width of apperture. 
- **max_cell_size:** Maximum size of mesh. 
- **f:** Operating frequency.
- **CST_obliqueCube.txt:** Simulation results of CST Studio Suite for a PEC cube under oblique incidence.  


## Outputs:
- **oblique_cube_30degree.pdf:** RCS comparison of MECA and CST Studio Suite for cube example.
- **two_parallel_plates_30degree.pdf:** RCS comparison of MECA and analytical expression for parallel plate example.

# Usage:
This program was designed to provide a fast numerical method to calculate the RCS of a structure using Modified Current Approximation (MECA) method. 

# Software Version Information:
**Python 3.11.4**

Libraries used in Python:
   * gmsh	        4.11.1
   * pip		23.2.1
   * numpy		1.24.3
   * matplotlib	        3.7.1


# References:
```
[1] J. G. Meana, J. A. Martinez-Lorenzo, F. Las-Heras, and C. Rappaport, “Wave scattering by dielectric and lossy materials using the modified equivalent current approximation (MECA),” IEEE Trans. Antennas Propag., vol. 58, no. 11, pp. 3757–3761, Nov. 2010.

[2] A. A. Kishk, Electromagnetic Waves Propagation in Complex Matter, 1st ed. Rijeka, Croatia: InTech, 2011.

[3] C. Geuzaine and J.-F. Remacle. "Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), pp. 1309-1331, 2009.

[4] Gmsh website: https://gmsh.info

[5] CST-Computer Simulation Technology AG, “CST studio suite 2022,” 2022. Accessed: Apr. 27, 2023. [Online]. Available: https://www.3ds.com/products-services/simulia/products/cst-studio-suite/

[6] R. Choupanzadeh and A. Zadehgol, "A Deep Neural Network Modeling Methodology for Efficient EMC Assessment of Shielding Enclosures Using MECA-Generated RCS Training Data," IEEE Transactions on Electromagnetic Compatibility, DOI: 10.1109/TEMC.2023.3316916.
 
[7] C. A. Balanis, Advanced Engineering Electromagnetics, 2nd ed. Hoboken, NJ, USA: Wiley, 2012.

[8] C. A. Balanis, Antenna Theory: Analysis and Design, 4th ed. Hoboken, NJ, USA: Wiley, 2016.

```
