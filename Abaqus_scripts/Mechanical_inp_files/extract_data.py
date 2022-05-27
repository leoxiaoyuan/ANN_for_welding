# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

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
    for index, node in enumerate(odb.rootAssembly.nodeSets['BD2_line'].nodes[0]):
        node_index.append((index, node.coordinates[0],node.coordinates[2]))

    xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
        'S', INTEGRATION_POINT, ((COMPONENT, 'S11'), )), ), nodeSets=("BD2_line", ))

    target_stress_data = []
    for index,stress_data in enumerate(xyList):
        for i in node_index:
            if index == i[0]:
                target_stress_data.append([i[0],i[1], i[2], stress_data[-1][-1]])
    zval = []
    result = []
    if target_stress_data[0][1] == 0:
        target_stress_data = sorted(target_stress_data, key=lambda x:x[2])
        for items in target_stress_data:
            zval.append(items[1]*1000)
            result.append(items[2]/1000000)
    else:
        target_stress_data = sorted(target_stress_data, key=lambda x:(x[2], x[1]))
        for index in range(len(target_stress_data)/2):
            zval.append(target_stress_data[index*2][2]*1000)
            sval = ((0 - target_stress_data[index*2+1][1])*(target_stress_data[index*2][-1]
                - target_stress_data[index*2+1][-1]) / (target_stress_data[index*2][1]
                - target_stress_data[index*2+1][1])) + target_stress_data[index*2+1][-1]
            result.append(sval/1000000)
    return zval, result

def get_D2_stress(S_field):
    node_index = []
    for index, node in enumerate(odb.rootAssembly.nodeSets['D2_line'].nodes[0]):
        node_index.append((index, node.coordinates[0], node.coordinates[2]))

    xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
        'S', INTEGRATION_POINT, ((COMPONENT, 'S11'), )), ), nodeSets=("D2_line", ))

    target_stress_data = []
    for index,stress_data in enumerate(xyList):
        for i in node_index:
            if index == i[0]:
                target_stress_data.append([i[0], i[1], i[2], stress_data[-1][-1]])
    xval = []
    result = []
    if target_stress_data[0][2] == 0.002:
        target_stress_data = sorted(target_stress_data, key=lambda x:x[1])
        for items in target_stress_data:
            xval.append(items[1]*1000)
            result.append(items[2]/1000000)
    else:
        target_stress_data = sorted(target_stress_data, key=lambda x:(x[1], x[2]))
        for index in range(len(target_stress_data)/2):
            xval.append(target_stress_data[index*2][1]*1000)
            sval = ((0.002 - target_stress_data[index*2+1][-2])*(target_stress_data[index*2][-1]
                - target_stress_data[index*2+1][-1]) / (target_stress_data[index*2][-2]
                - target_stress_data[index*2+1][-2])) + target_stress_data[index*2+1][-1]
            result.append(sval/1000000)
    return xval, result

# def get_mid_plane_node_set():
#     '''
#     get info and instanses for middle plane node set
    
#     '''
#     mid_plane_node_label_list = []
#     mid_plane_node_info = []
#     for node in odb.rootAssembly.nodeSets[' ALL NODES'].nodes[0]:
         
#         if  node.coordinates[0] == 0:
#             mid_plane_node_label_list.append(node.label)
#             mid_plane_node_info.append([node.label, node.coordinates[0], node.coordinates[1], node.coordinates[2]])
#     if 'mid_plane' in odb.rootAssembly.nodeSets:
#         mid_plane_node_set = odb.rootAssembly.nodeSets['mid_plane']
#     else:
#         mid_plane_node_set = odb.rootAssembly.NodeSetFromNodeLabels(
#                 name='mid_plane', nodeLabels= (('PART-ASSEMBLY', mid_plane_node_label_list),))
    
#     return mid_plane_node_set, mid_plane_node_info

def create_BD_line_set():
    '''
    get info and instanses for BD line node set
    
    '''
    BD_line_label_list = []
    BD_line_info = []
    for node in odb.rootAssembly.nodeSets[' ALL NODES'].nodes[0]:   
        if  node.coordinates[1] == 0:    
            if  abs(node.coordinates[0] - 0) < 0.0012:
                BD_line_label_list.append(node.label)
                BD_line_info.append([node.label, node.coordinates[0], node.coordinates[1], node.coordinates[2]])
    if 'BD2_line' in odb.rootAssembly.nodeSets:
        BD_line_set = odb.rootAssembly.nodeSets['BD2_line']
    else:
        BD_line_set = odb.rootAssembly.NodeSetFromNodeLabels(name='BD2_line', nodeLabels= (('PART-ASSEMBLY', BD_line_label_list),))
    return BD_line_set, BD_line_info

def create_D2_line_set():
    '''
    get info and instanses for BD line node set
    
    '''
    D2_line_label_list = []
    D2_line_info = []
    for node in odb.rootAssembly.nodeSets[' ALL NODES'].nodes[0]:   
        if  node.coordinates[1] == 0:    
            if abs(node.coordinates[2] - 0.002) < 0.0006:
                D2_line_label_list.append(node.label)
                D2_line_info.append([node.label, node.coordinates[0], node.coordinates[1], node.coordinates[2]])
    print(D2_line_info)
    if 'D2_line' in odb.rootAssembly.nodeSets:
        D2_line_set = odb.rootAssembly.nodeSets['D2_line']
    else:
        D2_line_set = odb.rootAssembly.NodeSetFromNodeLabels(name='D2_line', nodeLabels= (('PART-ASSEMBLY', D2_line_label_list),))
    return D2_line_set, D2_line_info

# def get_temp_field_his(node_set,step_name):
#     '''
#     get the history of temperature history of a node set
#     temp_field_his is the history of temperature
#     '''
#     temp_field_his = []
#     for step in step_name:
#         for frame in odb.steps[step].frames:
#             node_set_T = frame.fieldOutputs['NT11'].getSubset(region=node_set)
#             # temp_field_his.append([frame.description, node_set_T])
#             for values in node_set_T.values:
#                 temp_field_his.append(values.data)
#     return temp_field_his

# def get_frame_temp_field(frame_temp_field, node_info):
#     '''
#     get info for temperature field of a certain frame
#     frame_temp_field is the temperature field data
#     mid_plane_node_info is the node info
#     node_temperature_list [nodelabel node_x node_y node_z temperature]
#     '''
#     node_temperature_list = []
#     for index, values in enumerate(frame_temp_field.values):
#         node_temperature_list.append([node_info[index][0], node_info[index][1], 
#                     node_info[index][2], node_info[index][3], values.data])
#     return node_temperature_list
        

# def get_bead_shape(temperature_field, w_o_d, melting_temp):
#     '''
#     get bead width and depth using the temperature field data
#     w_o_d width dirction or depth direction
#     return bead width or bead depth
#     '''
        
#     #collect node postion and temperature for nodes which meet requirement
#     node_temperature_list_check = [[a,b,c,d,e] for [a,b,c,d,e] in temperature_field if e > melting_temp]
#     if not node_temperature_list_check:
#         return 0
#     else:
#         if w_o_d == 'width':
#             W_O_D = 2
#             node_temperature_list_filtered = [[a,b,c,d,e] for [a,b,c,d,e] in temperature_field if d == 0]
#         if w_o_d == 'depth':
#             W_O_D = 3    
#             node_temperature_list_filtered = [[a,b,c,d,e] for [a,b,c,d,e] in temperature_field if c == 0]
#         node_temperature_list_filtered = sorted(node_temperature_list_filtered, key=lambda x:x[W_O_D])
#         # print(node_temperature_list_filtered)
#         for index, items in enumerate(node_temperature_list_filtered):
#             if items[4] < melting_temp:
#                 node_1 = items
#                 node_2 = node_temperature_list_filtered[index-1]
#                 temp_1 = node_1[4]
#                 temp_2 = node_2[4]
#                 extra_length = (node_1[W_O_D]-node_2[W_O_D])*(temp_2-1400)/(temp_2-temp_1)
#                 totol_length = node_2[W_O_D] + extra_length
#                 break
#         return totol_length



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
# script_dir = os.getcwd()
script_dir = '/mnt/iusers01/mace01/q87448zm/scratch/data_generation/Mechanical_analysis'
os.chdir(script_dir)
obd_folder_root = script_dir
odb_root_list = findAllFilesWithSpecifiedSuffix(obd_folder_root, "odb")
for odb_root in odb_root_list:
    odb = session.openOdb(name=odb_root, readOnly=False) 
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)

    # Get fusion region shape
    # mid_plane_node_set, mid_plane_node_info = get_mid_plane_node_set()
    # mid_plane_temp_field_his = get_temp_field_his(mid_plane_node_set, 'Welding')
    # bead_shape = 0
    # for frames in mid_plane_temp_field_his:   
    #     temp_field = get_frame_temp_field(frames[1], mid_plane_node_info)
    #     melted_region_width = get_bead_shape(temp_field, 'width', 1400)
    #     melted_region_depth = get_bead_shape(temp_field, 'depth', 1400)
    #     if melted_region_width + melted_region_depth > bead_shape:
    #         bead_shape = melted_region_width + melted_region_depth
    #         bead_width = melted_region_width
    #         bead_depth = melted_region_depth
    #         target_temp_field = temp_field

    #Get stress field for BD
    S11_of_BD = []
    S11_of_D2 = []
    if len(odb.steps["Cooling_2"].frames):
        if odb.steps["Cooling_2"].frames[-1].frameValue == 1000:
            Frame=odb.steps["Cooling_2"].frames[-1]
            S_field=Frame.fieldOutputs['S']
            BD_line_set, BD_line_info = create_BD_line_set()
            BD_position, S11_of_BD = get_BD_stress(S_field)
            # D2_line_set, D2_line_info = create_D2_line_set()
            # D2_position, S11_of_D2 = get_D2_stress(S_field)
        else:
            S11_of_BD = ['Wrong']
            S11_of_D2 = ['Wrong']

    else:
        S11_of_BD = ['Wrong']
        S11_of_D2 = ['Wrong']

    #Get temperature field for BD
    # BD_temp_field_welding = get_temp_field_his(BD_line_set,['Welding'])
    # BD_temp_field_cooling = get_temp_field_his(BD_line_set,['Cooling_1','Cooling_2'])
    info = odb_root.split('/')[-1][4:-4]
    #Get total time
    # step_time_welding = []
    # step_time_cooling = []
    # for frames in odb.steps['Welding'].frames:
    #     step_time_welding.append(frames.frameValue)
    # for frames in odb.steps['Cooling_1'].frames:
    #     step_time_cooling.append(frames.frameValue)
    # for frames in odb.steps['Cooling_2'].frames:
    #     step_time_cooling.append(frames.frameValue)
    # with open(info + '_stress_field.csv', 'a+') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(target_temp_field)
    # for stepName in odb.steps.keys():
    #     print(stepName)
    with open('S11_along_BD.csv', 'a+') as f:
        writer = csv.writer(f)
        # BD_position = [' '] + BD_position
        # writer.writerow(BD_position)
        results = [info] + S11_of_BD
        writer.writerow(results)
        # D2_position = [' '] + D2_position
        # writer.writerow(D2_position)
        # results = [info] + S11_of_D2
        # writer.writerow(results)
    # with open('NT11_along_BD.csv', 'a+') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([info]) 
    #     writer.writerow(step_time_welding)
    #     writer.writerow(step_time_cooling)
    #     writer.writerow(BD_temp_field_welding) 
    #     writer.writerow(BD_temp_field_cooling)
    #     writer.writerow(S11_of_BD) 