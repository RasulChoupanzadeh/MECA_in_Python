
""" gmsh_parallel_plates.py     => This script generates meshes for two parallel plates using gsmh method desribed in the refrences. An application of this script can be found in [3].

Author: Rasul Choupanzadeh 
Date: 08/13/2023

Acknowledgement 1:
This project is completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, in the Applied Computational Electromagnetics and Signal/Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA.

Acknowledgement 2: This script is uses an open source 3D finite element mesh generator; which is called gmsh [1]. Further information is available in [2].


[1]  C. Geuzaine and J.-F. Remacle. "Gmsh: a three-dimensional finite element mesh generator with built-in pre- and
     post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), pp. 1309-1331, 2009.

[2]  https://gmsh.info

[3]  R. Choupanzadeh and A. Zadehgol, "A Deep Neural Network Modeling Methodology for Efficient EMC Assessment of Shielding Enclosures
     Using MECA-Generated RCS Training Data," IEEE Transactions on Electromagnetic Compatibility, DOI: 10.1109/TEMC.2023.3316916.

"""


## Input: x_dim, y_dim, h, l_app, w_app, max_cell_size, aperture              Output: num_facets, mesh_data, bary_data



# Import modules:
import gmsh
import sys
import numpy as np

# Initialize gmsh:
gmsh.initialize()


# Next we add a new model named "t1" (if gmsh.model.add() is not called a new unnamed model will be created on the fly, if necessary):
gmsh.model.add("t1")


## Input arguments from MECA.py
#x_dim = 5lambda
#y_dim = 5lambda
#h = 5lambda
#l_app = x_dim / 2       # x-direction aperture length
#w_app = y_dim / 5       # y-direction aperture length
#max_cell_size = lambda1 / 2


# cube points:
lc = max_cell_size 
point1 = gmsh.model.geo.add_point(-x_dim/2, -y_dim/2, -h/2, lc)                          ## or point1 = gmsh.model.geo.addPoint(0, 0, 0, lc)   
point2 = gmsh.model.geo.add_point(x_dim/2, -y_dim/2, -h/2, lc)
point3 = gmsh.model.geo.add_point(x_dim/2, y_dim/2, -h/2, lc)
point4 = gmsh.model.geo.add_point(-x_dim/2, y_dim/2, -h/2, lc)
point5 = gmsh.model.geo.add_point(-x_dim/2, y_dim/2, h/2, lc)
point6 = gmsh.model.geo.add_point(-x_dim/2, -y_dim/2, h/2, lc)
point7 = gmsh.model.geo.add_point(x_dim/2, -y_dim/2, h/2, lc)
point8 = gmsh.model.geo.add_point(x_dim/2, y_dim/2, h/2, lc)

# Edge of cube:
line1 = gmsh.model.geo.add_line(point1, point2)						## or line1 = gmsh.model.geo.addLine(point1, point2)
line2 = gmsh.model.geo.add_line(point2, point3)
line3 = gmsh.model.geo.add_line(point3, point4)
line4 = gmsh.model.geo.add_line(point4, point1)
line5 = gmsh.model.geo.add_line(point5, point6)
line6 = gmsh.model.geo.add_line(point6, point7)
line7 = gmsh.model.geo.add_line(point7, point8)
line8 = gmsh.model.geo.add_line(point8, point5)


# faces of cube:
curve1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])     		## or face1 = gmsh.model.geo.addCurveLoop([line1, line2, line3, line4])
curve2 = gmsh.model.geo.add_curve_loop([line5, line6, line7, line8])



# surfaces of cube:
surface1 = gmsh.model.geo.add_plane_surface([curve1])	
surface2 = gmsh.model.geo.add_plane_surface([curve2])


# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()

# Generate mesh:
gmsh.model.mesh.generate()


# Creates graphical user interface
if 'close' not in sys.argv:
	gmsh.fltk.run()


############################################ Data preparation ##################################################

# Find the elementType (it is always equal to 2 for 3-node triangles)  
elementType = gmsh.model.mesh.getElementType("Triangle", 1)               # ("Triangle", 1)=3-node triangle ==> elementTpye =2

# Find the nodes (with corrdinations) of specific elementType (2 (triangles) in this case) with element tag number=tag,  tag < 0 : means for all elements,   the function returns `nodeTags', `coord', `parametricCoord'
ElementNodes = gmsh.model.mesh.getNodesByElementType(elementType, tag=-1, returnParametricCoord=True)      # ElementNodes_coord =  ElementNodes[1]  ==> Vector of coordinates of All nodes of triangle elements
num_Nodes =  np.max(ElementNodes[0])									   # Number of total nodes
Node_dim =  int(len(ElementNodes[1]) / len(ElementNodes[0]))						   # Dimension of each Node

# Find the elementTags with corresponding `nodeTags'.
elementNode_tags = gmsh.model.mesh.getElementsByType(elementType, tag=-1, task=0, numTasks=1)
element_tags = elementNode_tags[0]
num_Elements =  len(element_tags)


# Property of `elementType', such as `elementName', `dim', `order', `numNodes', `localNodeCoord', `numPrimaryNodes'.
ElementProperty = gmsh.model.mesh.getElementProperties(elementType)		
elementType_num_nodes = ElementProperty[3]    					# Number of nodes connected to each element (it is 3 for triangle element)


mesh_data = np.zeros(shape=(num_Elements, elementType_num_nodes, Node_dim))
# This following only for elements with 3 nodes, for quadrangle elements it must be modified by adding n4
i = 0
for el_tag in element_tags:
	n1 = gmsh.model.mesh.getElement(el_tag)[1][0]           # nodeTag of 1st node
	n2 = gmsh.model.mesh.getElement(el_tag)[1][1]		# nodeTag of 2nd node
	n3 = gmsh.model.mesh.getElement(el_tag)[1][2]		# nodeTag of 3rd node
	mesh_data[i] = np.array([gmsh.model.mesh.getNode(n1)[0],gmsh.model.mesh.getNode(n2)[0],gmsh.model.mesh.getNode(n3)[0]])     # save the coordinates of nodeTags
	i = i + 1


############################################ barycenter data ###################################################
# Find barycenters of all elements of type `elementType' 
# gmsh.model.mesh.getBarycenters(elementType, tag, fast, primary, task=0, numTasks=1)
# If `primary' is set, only the primary nodes of the elements are taken into account for the barycenter calculation.
# If `fast'is set, the function returns the sum of the primary node coordinates without normalizing by the number of nodes.
# If `numTasks' > 1, only compute and return the part of the data indexed by `task'.
barycenters = gmsh.model.mesh.getBarycenters(elementType, -1, 0, 1)
bary_data = barycenters.reshape(num_Elements, Node_dim)

################################################################################################################

#This should be called when you are done using the Gmsh Python API:
gmsh.finalize()

print("Mesh Info:", num_Nodes, "nodes and ", num_Elements, "triangular elements")


