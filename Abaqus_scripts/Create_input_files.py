# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import os 
import sys
import numpy as np 
from collections import OrderedDict
def create_model(specimen_length, specimen_width, specimen_hight, bead_length, arc_speed, heat_input, heat_source_size_coeff=1):
    room_temperature = 20
    stop_time_at_start = 0
    stop_time_at_end = 0
    finest_mesh_size = 0.0015
    # Modeling
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.1)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.sketchOptions.setValues(decimalPlaces=3)
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(-specimen_length/2, 0.0), point2=(specimen_length/2, specimen_width/2))
    p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['Part-1']
    c = p.cells
    e = p.edges
    d = p.datums
    p.BaseSolidExtrude(sketch=s, depth=specimen_hight)

    # Partition
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.02)
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=0.01)
    # p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=bead_length/2)
    # p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-bead_length/2)
    p.PartitionCellByDatumPlane(datumPlane=d[2], cells=c)
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=c)
    # p.PartitionCellByDatumPlane(datumPlane=d[4], cells=c)
    # p.PartitionCellByDatumPlane(datumPlane=d[5], cells=c)
    # Property definition
    mdb.models['Model-1'].Material(name='AISI316L')
    mdb.models['Model-1'].materials['AISI316L'].Density(table=((7966.0, ), ))
    mdb.models['Model-1'].materials['AISI316L'].Elastic(temperatureDependency=ON, 
        table=((195600000000, 0.294, 20.0), (191200000000, 0.294, 100.0), (
        185700000000, 0.294, 200.0), (179600000000, 0.294, 300.0), (
        172600000000, 0.294, 400.0), (164500000000, 0.294, 500.0), (
        155000000000, 0.294, 600.0), (144100000000, 0.294, 700.0), (
        131400000000, 0.294, 800.0), (116800000000, 0.294, 900.0), (
        100000000000, 0.294, 1000.0), (80000000000, 0.294, 1100.0), (
        57000000000, 0.294, 1200.0), (30000000000, 0.294, 1300.0), (2000000000, 0.294, 1400.0)))
    mdb.models['Model-1'].materials['AISI316L'].Plastic(temperatureDependency=ON, 
        table=((269119973.6, 0.0, 0.0), (248993663.6, 0.0, 100.0), (
        228867353.5, 0.0, 200.0), (208741043.4, 0.0, 300.0), (188614733.4, 0.0, 
        400.0), (168488423.3, 0.0, 500.0), (148362113.3, 0.0, 600.0), (
        128235803.2, 0.0, 700.0), (108109493.1, 0.0, 800.0), (87983183.08, 0.0, 
        900.0)))
    mdb.models['Model-1'].materials['AISI316L'].SpecificHeat(
        temperatureDependency=ON, table=((492, 20.0), (502, 
        100.0), (514, 200.0), (526, 300.0), (538, 
        400.0), (550, 500.0), (562, 600.0), (575, 
        700.0), (587, 800.0), (599, 900.0), (611,
        1000.0), (623, 1100.0), (635, 1200), (647, 1300.0), (659, 1400.0)))
    mdb.models['Model-1'].materials['AISI316L'].Conductivity(
        temperatureDependency=ON, table=((14.12, 20.0), (15.26, 
        100.0), (16.69, 200.0), (18.11, 300.0), (19.54, 
        400.0), (20.96, 500.0), (22.38, 600.0), (23.81, 
        700.0), (25.23, 800.0), (26.66, 900.0), (28.08, 
        1000.0), (29.50, 1100.0), (30.93, 1200.0), (32.35, 1300.0), (33.78, 1400.0)))
    mdb.models['Model-1'].materials['AISI316L'].Expansion(table=((1.456e-05, 20.0), 
        (1.539e-05, 100.0), (1.621e-05, 200.0), (1.686e-05, 300.0), (
        1.737e-05, 400.0), (1.778e-05, 500.0), (1.812e-05, 600.0), (
        1.843e-05, 700.0), (1.872e-05, 800.0), (1.899e-05, 900.0), (
        1.927e-05, 1000.0), (1.953e-05, 1100.0), (1.979e-05, 1200.0), (
        2.002e-05, 1300.0), (2.021e-05, 1400.0)), 
        temperatureDependency=ON)  
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
        material='AISI316L', thickness=None)

    region = regionToolset.Region(cells=c)
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

    #Step
    #decrease the time incr
    mdb.models['Model-1'].CoupledTempDisplacementStep(name='Welding', 
        previous='Initial', timePeriod= bead_length/arc_speed + 
        stop_time_at_start + stop_time_at_end, maxNumInc=1000000000, 
        timeIncrementationMethod=FIXED, initialInc=0.1, deltmx=None, 
        cetol=None, creepIntegration=None, nlgeom=ON)
    # mdb.models['Model-1'].CoupledTempDisplacementStep(name='Welding', 
    #     previous='Initial', timePeriod= bead_length/arc_speed + 
    #     stop_time_at_start + stop_time_at_end, initialInc=0.5,
    #     minInc=0.01, maxInc=10, deltmx=1000000000.0, 
    #     cetol=None, creepIntegration=None, nlgeom=ON)
    ##ON(LARGE DISPLACEMENT) OR OFF(SMALL))?
    mdb.models['Model-1'].CoupledTempDisplacementStep(name='Cooling', 
        previous='Welding', timePeriod=1200, initialInc=10.0,
        minInc=0.1, maxInc=25.0, deltmx=10000000.0,nlgeom=ON)

    ##initialInc may have problem

    # Assembly
    a = mdb.models['Model-1'].rootAssembly
    myinstance = a.Instance(name='Part-assembly', part=p, dependent=ON)
    c1 = a.instances['Part-assembly'].cells
    f1 = a.instances['Part-assembly'].faces
    e1 = a.instances['Part-assembly'].edges
    v1 = a.instances['Part-assembly'].vertices
    n1 = a.instances['Part-assembly'].nodes

    # Interaciton
    region=regionToolset.Region(side1Faces=f1)
    mdb.models['Model-1'].FilmCondition(name='Surface film condition', 
        createStepName='Welding', surface=region, definition=EMBEDDED_COEFF, 
        filmCoeff=15, filmCoeffAmplitude='', sinkTemperature = room_temperature, 
        sinkAmplitude='', sinkDistributionType=UNIFORM, sinkFieldName='')
    mdb.models['Model-1'].RadiationToAmbient(name='Surface radiation', 
        createStepName='Welding', surface=region, radiationType=AMBIENT, 
        distributionType=UNIFORM, field='', emissivity=0.7, 
        ambientTemperature = room_temperature, ambientTemperatureAmp='')

    # Attribute
    mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-11)

    # Load
    # load_cell = c1.findAt(((0,0,0),))
    region = regionToolset.Region(cells=c1)
    mdb.models['Model-1'].BodyHeatFlux(name='Load-1', createStepName='Welding', 
        region=region, distributionType=USER_DEFINED)
    mdb.models['Model-1'].loads['Load-1'].deactivate('Cooling')

    # Boundary Condition
    left_vertice = v1.findAt(((specimen_length/2,specimen_width/2,specimen_hight),))
    right_vertice = v1.findAt(((-specimen_length/2,specimen_width/2,specimen_hight),))
    region = a.Set(vertices=left_vertice, name='left_vertice')
    mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
        region=region, u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    region = a.Set(vertices=right_vertice, name='right_vertice')
    mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial', 
        region=region, u1=UNSET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)

    front_face_1_center = (0,0,specimen_hight/2)
    front_face_1 = f1.findAt((front_face_1_center,))
    region = regionToolset.Region(faces=front_face_1)
    mdb.models['Model-1'].YsymmBC(name='BC-3', createStepName='Initial', 
        region=region, localCsys=None)

    # Predefined field
    region = regionToolset.Region(vertices=v1, edges=e1, faces=f1, 
            cells=c1)
    mdb.models['Model-1'].Temperature(name='Predefined Field-1', 
        createStepName='Initial', region=region, distributionType=UNIFORM, 
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(room_temperature, 
        ))

    # Mesh
    pickedEdges = e.findAt(((0,0,0),))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size*2, deviationFactor=0.1, 
        constraint=FINER)
    pickedEdges = e.findAt(((specimen_length/2,0,0.0025),))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size, deviationFactor=0.1, 
        constraint=FINER)
    pickedEdges = e.findAt(((specimen_length/2,0.002,0),))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size, deviationFactor=0.1, 
        constraint=FINER)
    p.seedPart(size=finest_mesh_size*3, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()

    elemType1 = mesh.ElemType(elemCode=C3D8T, elemLibrary=STANDARD, 
        secondOrderAccuracy=OFF, distortionControl=DEFAULT)
    p.setElementType(regions=(c, ), elemTypes=(elemType1,))

    # Output
    mdb.models['Model-1'].FieldOutputRequest(name='F-Output-1', 
        createStepName='Welding', variables=('S', 'U', 'NT', ))
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()
    target_edge_node = e1.findAt(((0,0,0),))[0].getNodes()[2]
    target_edge = target_edge_node.getNodesByFeatureEdge(20)
    target_region = a.Set(nodes=target_edge, name='target_edge_nodes')

    top_face_1 = f1.findAt(((0,finest_mesh_size,0), ))
    top_face_2 = f1.findAt(((0, 0.06, 0), ))
    target_region_2 = a.Set(faces=top_face_1 + top_face_2, name='top_face')
    top_face_nodes = target_region_2.nodes
    target_region = a.Set(nodes=top_face_nodes, name='top_face_nodes')

    front_face_1 = f1.findAt(((0,0,finest_mesh_size), ))
    front_face_2 = f1.findAt(((0, 0, 0.017-finest_mesh_size), ))
    target_region_2 = a.Set(faces=front_face_1 + front_face_2, name='front_face')
    front_face_nodes = target_region_2.nodes
    target_region = a.Set(nodes=front_face_nodes, name='front_face_nodes')

    mdb.models['Model-1'].FieldOutputRequest(name='F-Output-2', 
        createStepName='Cooling', variables=('S',), frequency=LAST_INCREMENT, 
        region=target_region, sectionPoints=DEFAULT, rebar=EXCLUDE)

    # create inp
    name = 'job-'+ str(int(specimen_length*1000)) +'-'+ str(int(specimen_width*1000)) + '-' + str(
        int(specimen_hight*1000)) + '-' + str(int(bead_length*1000)) + '-'+ str(
        int(arc_speed*1000000)) + '-' + str(int(heat_input)) + '-'+ str(int(heat_source_size_coeff)) 

    # UFLUX_script=''
    # with open('H:/PhD_data/UFLUX/UFLUX_Double_ellipsoid_heatsource2.for','r+') as fr:
    #     for line in fr:
    #         if 'bead_length =' in line:
    #             line = '      bead_length = ' + str(bead_length) + '\n'
    #         if 'stop_time_at_start =' in line:
    #             line = '      stop_time_at_start = ' + str(stop_time_at_start) + '\n'
    #         if 'welding_time =' in line:
    #             line = '      welding_time = ' + str(bead_length/arc_speed) + '\n'
    #         if 'arc_speed =' in line:
    #             line = '      arc_speed = ' + str(arc_speed) + '\n'
    #         if 'heat_input =' in line:
    #             line = '      heat_input = ' + str(heat_input) + '\n'
    #         if 'heat_source_size_coeff =' in line:
    #             line = '      heat_source_size_coeff = ' + str(heat_source_size_coeff) + '\n'
    #         UFLUX_script+=line
    # fr.close()

    # with open('H:/PhD_data/UFLUX/UFLUX_Double_ellipsoid_heatsource2.for','w') as fr:
    #     fr.write(UFLUX_script)
    # fr.close()
                        
    # Uload_path = 'H:/PhD_data/UFLUX/UFLUX_Double_ellipsoid_heatsource2.for'
    mdb.Job(name=name, model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs[name].writeInput(consistencyChecking=OFF)
    # mdb.jobs[name].submit(consistencyChecking=OFF)

def create_arg_list(arg):
    # print(arg)
    l_limit = float(arg.split('-')[-3])
    u_limit = float(arg.split('-')[-2])
    num = int(arg.split('-')[-1])
    para_list = np.linspace(l_limit, u_limit, num)
    return para_list

# if __name__ == "__main__":
os.chdir("/mnt/iusers01/mace01/q87448zm/scratch/test3")
#init parameter dictionary
para_dic = OrderedDict()
para_dic['specimen_length'] = []
para_dic['specimen_width'] = []
para_dic['specimen_hight'] = []
para_dic['bead_length'] = []
para_dic['arc_speed'] = []
para_dic['heat_input'] = []
para_dic['heat_source_size_coeff'] = []
sys_argv = sys.argv[-7:]
for index, item in enumerate(para_dic):
    if '-' in sys_argv[index]:
        para_list = create_arg_list(sys_argv[index])
        para_dic[item] = para_list
    else:
        para_dic[item] = [float(sys_argv[index])]


for specimen_length in para_dic['specimen_length']:
    for specimen_width in para_dic['specimen_width']:
        for specimen_hight in para_dic['specimen_hight']:
            for bead_length in para_dic['bead_length']:
                for arc_speed in para_dic['arc_speed']:
                    for heat_input in para_dic['heat_input']:
                        for heat_source_size_coeff in para_dic['heat_source_size_coeff']: 
                            create_model(specimen_length,specimen_width,specimen_hight,bead_length,arc_speed,heat_input,heat_source_size_coeff)



