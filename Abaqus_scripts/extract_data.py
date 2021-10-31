# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import sys
# import section
# import regionToolset
# import displayGroupMdbToolset as dgm
# import part
# import material
# import assembly
# import step
# import interaction
# import load
# import mesh
# import optimization
# import job
# import sketch
# import visualization
import xyPlot
# import displayGroupOdbToolset as dgo
# import connectorBehavior
import csv
import os
def get_BD_stress(S_field):
    node_index = []
    for index, node in enumerate(odb.rootAssembly.nodeSets['FRONT_FACE_NODES'].nodes[0]):
        if node.coordinates[0] == 0:
            node_index.append((index, node.coordinates[2]))

    xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
        'S', INTEGRATION_POINT, ((COMPONENT, 'S11'), )), ), nodeSets=("FRONT_FACE_NODES", ))

    target_stress_data = []
    for index,stress_data in enumerate(xyList):
        for i in node_index:
            if index == i[0]:
                target_stress_data.append([i[0],i[1],stress_data[-1][-1]])

    target_stress_data = sorted(target_stress_data, key=lambda x:x[1])
    depth = []
    result = []
    for items in target_stress_data:
        depth.append(items[1])
        result.append(items[2])
    return depth, result
            
def get_bead_shape(face_name, temperature_field):
    '''
    get bead width and depth using the temperature field data
    get bead width using top face data
    get bead depth using front face data
    return bead width or bead depth
    '''
    node_temperature_list = []
    node_set = odb.rootAssembly.nodeSets[face_name]
    node_set_T = temperature_field.getSubset(region=node_set)
    #collect node label and temperatue for all nodes on face
    for T in node_set_T.values:
        node_position = odb.rootAssembly.nodeSets[' ALL NODES'].nodes[0][T.nodeLabel-1].coordinates #The label differs from the serial number by one#      
        node_temperature_list.append((T.nodeLabel,node_position,T.data))
        
    #collect node postion and temperature for nodes which meet requirement
    node_temperature_list_filtered = [(x,y,z) for (x,y,z) in node_temperature_list if z > 1450]
    if not node_temperature_list_filtered:
        return 0
    #Select edge node whose temperature is above melting temperature 1450
    if face_name == 'TOP_FACE_NODES':
        target_node = max(node_temperature_list_filtered, key=lambda x: x[1][1])
        bead_width_list_for_checking = []
        for items in node_temperature_list_filtered:# many maximum values case
            if items[1][1] == target_node[1][1]:
                #find an adjacent node
                target_node_nearest =  find_nearest_node_label(items, 1)
                #find the temperature of the target node and adjacent node 
                node1_T = target_node[2]
                for items in node_temperature_list:
                    if items[0] == target_node_nearest:
                        node2_T = items[2]
                extra_width = finest_mesh_size*(node1_T-1450)/(node1_T-node2_T)
                bead_width = target_node[1][1] + extra_width
                bead_width_list_for_checking.append(bead_width)
        bead_width = max(bead_width_list_for_checking)
        return bead_width       


    if face_name == 'FRONT_FACE_NODES':
        target_node = max(node_temperature_list_filtered, key=lambda x: x[1][2])
        bead_depth_list_for_checking = []
        for items in node_temperature_list_filtered:
            if items[1][2] == target_node[1][2]:
                #find an adjacent node
                target_node_nearest =  find_nearest_node_label(items, 2)
                #find the temperature of the target node and adjacent node 
                node1_T = target_node[2]
                for items in node_temperature_list:
                    if items[0] == target_node_nearest:
                        node2_T = items[2]
                extra_depth = finest_mesh_size*(node1_T-1450)/(node1_T-node2_T)
                bead_depth = target_node[1][2] + extra_depth
                bead_depth_list_for_checking.append(bead_depth)

        bead_depth = max(bead_depth_list_for_checking)
        return bead_depth
    # return bead_width, bead_depth

def find_nearest_node_label(target_node, direction):
        '''
        find an adjacent node
        if searching nearest_node in top face, direction = 1
        if searching nearest_node in front face, direction = 2
        '''
        if direction == 1:
            top_face_node_index = nearestNodeModule.findNearestNode(xcoord=target_node[1][0], 
                                                ycoord=target_node[1][1]+finest_mesh_size, 
                                zcoord=target_node[1][2], name=odb_root, instanceName='')[0]  
        if direction == 2:
            top_face_node_index = nearestNodeModule.findNearestNode(xcoord=target_node[1][0], 
                                ycoord=target_node[1][1], zcoord=target_node[1][2]
                                + finest_mesh_size, name=odb_root, instanceName='')[0] 
        
        return top_face_node_index



def findAllFilesWithSpecifiedSuffix(target_dir, target_suffix="odb"):
    '''
    find all obd files in the target folder
    '''
    find_res = []
    target_suffix_dot = "." + target_suffix
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == target_suffix_dot:
                find_res.append(os.path.join(root_path, file))
    return find_res

#open database
script_dir = os.getcwd()
os.chdir(script_dir)
obd_folder_root = script_dir
odb_root_list = findAllFilesWithSpecifiedSuffix(obd_folder_root, "odb")
for odb_root in odb_root_list:
    finest_mesh_size = 0.001
    odb = session.openOdb(name=odb_root, readOnly=False) 
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)

    #collect S11 data
    # node_position = []
    # for node in odb.rootAssembly.nodeSets['TARGET_EDGE_NODES'].nodes[0]:
    #     node_position.append(node.coordinates[0])

    # xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    #     'S', INTEGRATION_POINT, ((COMPONENT, 'S11'), )), ), nodeSets=("TARGET_EDGE_NODES", ))
    # target_stress_data = []
    # for index,stress_data in enumerate(xyList):
    #     target_stress_data.append(stress_data[-1][-1])
        
    #collect bead width data
    #load nearestNodeModule plug-in
    sys.path.insert(10, '/opt/apps/apps/binapps/abaqus/2020/SIMULIA/EstProducts/2020/linux_a64/code/python2.7/lib/abaqus_plugins/findNearestNode')
    import nearestNodeModule
    # top_face_node_index = nearestNodeModule.findNearestNode(xcoord=0, ycoord=0.1, 
    #     zcoord=0, name='H:/PhD_data/inp_files/TEST1.odb', instanceName='')[0]
    # top_face_node = odb.rootAssembly.nodeSets[' ALL NODES'].nodes[0][top_face_node_index]
    # top_face_nodes = odb.rootAssembly.nodeSets[' ALL NODES'].getNodesByFaceAngle(20)
    # target_region = odb.rootAssembly.NodeSet(nodes=top_face_nodes, name='top_face_nodes')
    # f1 = odb.rootAssembly.instances['PART-ASSEMBLY'].faces
    # top_face = f1.findAt(((0,1,0),))
    # target_region = odb.rootAssembly.NodeSet(faces=top_face, name='top_face')
    Frame=odb.steps["Welding"].frames[-1]
    temperature_field=Frame.fieldOutputs['NT11']
    Frame=odb.steps["Cooling"].frames[-1]
    S_field=Frame.fieldOutputs['S']
    # top_face=odb.rootAssembly.nodeSets['TOP_FACE_NODES']
    # top_face_T= temperature_field.getSubset(region=top_face)
    # front_face=odb.rootAssembly.nodeSets['FRONT_FACE_NODES']
    # front_face_T= temperature_field.getSubset(region=front_face)

    bead_width = get_bead_shape('TOP_FACE_NODES',temperature_field)
    bead_depth = get_bead_shape('FRONT_FACE_NODES',temperature_field)
    BD_position, S11_of_BD = get_BD_stress(S_field)

    # create and write to a file
    with open('S11_along_BD.csv', 'a+') as f: # 采用b的方式处理可以省去很多问题
        writer = csv.writer(f)
        info = odb_root.split('/')[-1][4:-4]
        BD_position_z = [ '', 'bead_width', 'bead_depth'] + BD_position
        results = [info] + [bead_width] + [bead_depth] + S11_of_BD
        writer.writerow(BD_position_z)
        writer.writerow(results)
    



# file = open('results_file.txt', 'w') # create and write to a named file in your work directory
# file.write('Label \t\t S11 \n') # write first line for coloumn labeling - \t tab \n newline

# # go throug all stress values and write to the .txt file
# for S in RegioncareS.values:
#     file.write('%d \t\t %.1f \n' % (S.integrationPoint, S.data[0]))
# file.close()

# print(RegioncareS)
# name =str(int(bead_length)) + '  ' + str(
#     int(bead_width)) + '  ' + str(int(bead_hight)) + '  ' + str(int(heat_input)/1000000)
# session.XYDataFromPath(name=name, path=pth, includeIntersections=True, 
#     projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
#     projectionTolerance=0, shape=DEFORMED, labelType=TRUE_DISTANCE)
# x = session.xyDataObjects[name]
# session.writeXYReport(fileName= str(int(specimen_length*1000)) +'mm-'+ str(int(specimen_width*1000)) + 'mm-' + str(
#     int(specimen_hight*1000)) + 'mm-' + str(int(bead_length)) + 'mm-'+ str(
#     int(bead_width)) + 'mm-' + str(int(bead_hight)) + 'mm-'+ str(int(heat_input)) + '.txt', xyData=(x))
