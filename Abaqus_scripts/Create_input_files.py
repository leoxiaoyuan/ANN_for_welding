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
import csv

def create_thermal_analysis_model(specimen_length, specimen_width, specimen_hight, bead_length, arc_speed, heat_input):
    room_temperature = 20
    stop_time_at_start = 0
    stop_time_at_end = 0
    finest_mesh_size = 0.0006
    HAZ_length = bead_length + 0.02
    HAZ_width = 0.04
    HAZ_depth = 0.01

    # Modeling
    mdb.Model(name='Model-1', modelType=STANDARD_EXPLICIT)
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
    del mdb.models['Model-1'].sketches['__profile__']

    # Partition
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=HAZ_width/2)
    if specimen_hight > HAZ_depth:
        p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=HAZ_depth)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=HAZ_length/2)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-HAZ_length/2)
    p.PartitionCellByDatumPlane(datumPlane=d[2], cells=c)
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=c)
    p.PartitionCellByDatumPlane(datumPlane=d[4], cells=c)
    if specimen_hight > HAZ_depth:
        p.PartitionCellByDatumPlane(datumPlane=d[5], cells=c)

    # Property definition
    mdb.models['Model-1'].Material(name='AISI316L-parent')
    mdb.models['Model-1'].materials['AISI316L-parent'].Density(table=((7966.0, ), ))
    mdb.models['Model-1'].materials['AISI316L-parent'].SpecificHeat(
        temperatureDependency=ON, table=((492, 20.0), (502, 
        100.0), (514, 200.0), (526, 300.0), (538, 
        400.0), (550, 500.0), (562, 600.0), (575, 
        700.0), (587, 800.0), (599, 900.0), (611,
        1000.0), (623, 1100.0), (635, 1200), (647, 1300.0), (659, 1400.0)))
    mdb.models['Model-1'].materials['AISI316L-parent'].Conductivity(
        temperatureDependency=ON, table=((14.12, 20.0), (15.26, 
        100.0), (16.69, 200.0), (18.11, 300.0), (19.54, 
        400.0), (20.96, 500.0), (22.38, 600.0), (23.81, 
        700.0), (25.23, 800.0), (26.66, 900.0), (28.08, 
        1000.0), (29.50, 1100.0), (30.93, 1200.0), (32.35, 1300.0), (33.78, 1400.0)))
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
        material='AISI316L-parent', thickness=None)
    c = p.cells
    region = regionToolset.Region(cells=c)
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

    #Step
    #decrease the time incr
    mdb.models['Model-1'].HeatTransferStep(name='Welding', 
        previous='Initial', timePeriod= bead_length/arc_speed + 
        stop_time_at_start + stop_time_at_end, maxNumInc=1000000000, 
        timeIncrementationMethod=FIXED, initialInc=0.1)
    mdb.models['Model-1'].HeatTransferStep(name='Cooling_1', 
        previous='Welding', timePeriod=70, initialInc=0.1,maxNumInc=1000000000,
        minInc=0.05, maxInc=5.0, deltmx=10000000.0)
    mdb.models['Model-1'].HeatTransferStep(name='Cooling_2', 
        previous='Cooling_1', timePeriod=1000, initialInc=5,maxNumInc=1000000000,
        minInc=1, maxInc=50, deltmx=10000000.0)

    # Assembly
    a = mdb.models['Model-1'].rootAssembly
    myinstance = a.Instance(name='Part-assembly', part=p, dependent=ON)
    c1 = a.instances['Part-assembly'].cells
    f1 = a.instances['Part-assembly'].faces
    e1 = a.instances['Part-assembly'].edges
    v1 = a.instances['Part-assembly'].vertices
    n1 = a.instances['Part-assembly'].nodes

    # Interaciton
    face_1_node = (specimen_length/2, finest_mesh_size, finest_mesh_size)
    face_1 = f1.findAt((face_1_node,))[0]
    tar_face_1 = face_1.getFacesByFaceAngle(20)
    face_2_node = (-specimen_length/2,finest_mesh_size,finest_mesh_size)
    face_2 = f1.findAt((face_2_node,))[0]
    tar_face_2 = face_2.getFacesByFaceAngle(20)
    face_3_node = (0, finest_mesh_size, 0)
    face_3 = f1.findAt((face_3_node,))[0]
    tar_face_3 = face_3.getFacesByFaceAngle(20)
    face_4_node = (0, finest_mesh_size, specimen_hight)
    face_4 = f1.findAt((face_4_node,))[0]
    tar_face_4 = face_4.getFacesByFaceAngle(20)
    face_5_node = (0, specimen_width/2, finest_mesh_size)
    face_5 = f1.findAt((face_5_node,))[0]
    tar_face_5 = face_5.getFacesByFaceAngle(20)
    target_face = tar_face_1 + tar_face_2 + tar_face_3 + tar_face_4 + tar_face_5
    region=regionToolset.Region(side1Faces=target_face)
    mdb.models['Model-1'].FilmCondition(name='Surface film condition', 
        createStepName='Welding', surface=region, definition=EMBEDDED_COEFF, 
        filmCoeff=15, filmCoeffAmplitude='', sinkTemperature = room_temperature, 
        sinkAmplitude='', sinkDistributionType=UNIFORM, sinkFieldName='')
    mdb.models['Model-1'].RadiationToAmbient(name='Surface radiation', 
        createStepName='Welding', surface=region, radiationType=AMBIENT, 
        distributionType=UNIFORM, field='', emissivity=0.7, 
        ambientTemperature = room_temperature, ambientTemperatureAmp='')

    # Attribute
    mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-8)

    # Load
    region = regionToolset.Region(cells=c1)
    mdb.models['Model-1'].BodyHeatFlux(name='Load-1', createStepName='Welding', 
        region=region, distributionType=USER_DEFINED)
    mdb.models['Model-1'].loads['Load-1'].deactivate('Cooling_1')

    # Predefined field
    region = regionToolset.Region(vertices=v1, edges=e1, faces=f1, 
            cells=c1)
    mdb.models['Model-1'].Temperature(name='Predefined Field-1', 
        createStepName='Initial', region=region, distributionType=UNIFORM, 
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(room_temperature, 
        ))

    # Mesh
    #bead mesh
    p = mdb.models['Model-1'].parts['Part-1']
    e = p.edges
    p.seedPart(size=finest_mesh_size*4, deviationFactor=0.1, minSizeFactor=0.1)
    pickedEdges = (e.findAt(((0,0,0),))) 
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size*2, deviationFactor=0.1, minSizeFactor=0.1,
        constraint=FINER)
    pickedEdges = (e.findAt(((HAZ_length/2, finest_mesh_size ,0),)))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size*2, deviationFactor=0.1, minSizeFactor=0.1,
        constraint=FINER)
    pickedEdges = (e.findAt(((HAZ_length/2, 0,finest_mesh_size),)))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size, deviationFactor=0.1, minSizeFactor=0.1,
        constraint=FINER)
    p.generateMesh()    
    elemType1 = mesh.ElemType(elemCode=DC3D20, elemLibrary=STANDARD, 
        secondOrderAccuracy=OFF, distortionControl=DEFAULT)
    p = mdb.models['Model-1'].parts['Part-1']
    c = p.cells
    p.setElementType(regions=(c, ), elemTypes=(elemType1,))

    # Output
    mdb.models['Model-1'].FieldOutputRequest(name='F-Output-1', 
        createStepName='Welding', variables=('NT', ))
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()

    # Create inp
    name = 'T-'+ str(int(specimen_length*1000)) +'-'+ str(int(specimen_width*1000)) + '-' + str(
        int(specimen_hight*1000)) + '-' + str(int(bead_length*1000)) + '-'+ str(
        int(arc_speed*1000000)) + '-' + str(int(heat_input)) 
    mdb.Job(name=name, model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs[name].writeInput(consistencyChecking=OFF)

def create_Mechanical_analysis_model(specimen_length, specimen_width, specimen_hight, bead_length, arc_speed, heat_input):
    room_temperature = 20
    stop_time_at_start = 0
    stop_time_at_end = 0
    finest_mesh_size = 0.0006
    HAZ_length = bead_length + 0.02
    HAZ_width = 0.04
    HAZ_depth = 0.01

    # Modeling
    mdb.Model(name='Model-1', modelType=STANDARD_EXPLICIT)
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
    del mdb.models['Model-1'].sketches['__profile__']

    # Partition
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=HAZ_width/2)
    if specimen_hight > HAZ_depth:
        p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=HAZ_depth)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=HAZ_length/2)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-HAZ_length/2)
    p.PartitionCellByDatumPlane(datumPlane=d[2], cells=c)
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=c)
    p.PartitionCellByDatumPlane(datumPlane=d[4], cells=c)
    if specimen_hight > HAZ_depth:
        p.PartitionCellByDatumPlane(datumPlane=d[5], cells=c)

    # Property definition
    mdb.models['Model-1'].Material(name='AISI316L-parent')
    mdb.models['Model-1'].materials['AISI316L-parent'].Density(table=((7966.0, ), ))

    mdb.models['Model-1'].materials['AISI316L-parent'].Elastic(temperatureDependency=ON, 
        table=((195600000000, 0.294, 20.0), (191200000000, 0.294, 100.0), 
        (185700000000, 0.294, 200.0), (179600000000, 0.294, 300.0), 
        (172600000000, 0.294, 400.0), (164500000000, 0.294, 500.0), 
        (155000000000, 0.294, 600.0), (144100000000, 0.294, 700.0), 
        (131400000000, 0.294, 800.0), (116800000000, 0.294, 900.0), 
        (100000000000, 0.294, 1000.0), (80000000000, 0.294, 1100.0), 
        (57000000000, 0.294, 1200.0), (30000000000, 0.294, 1300.0), 
        (2000000000, 0.294, 1400.0),))

    mdb.models['Model-1'].materials['AISI316L-parent'].Plastic(hardening=COMBINED, 
        dataType=PARAMETERS, temperatureDependency=ON, numBackstresses=2, 
        table=( (125600000, 156435000000., 1410.85,   6134000000.,   47.19,     20.),
                ( 97600000, 100631000000., 1410.85,   5568000000.,   47.19,    275.),
                ( 90900000,  64341000000., 1410.85,   5227000000.,   47.19,    550.),
                ( 71400000,  56232000000., 1410.85,   4108000000.,   47.19,    750.),
                ( 66200000,  49588000000., 1410.85,    292000000.,   47.19,    900.),
                ( 31800000,      5000000., 1410.85,      5000000.,   47.19,   1000.),
                ( 25000000,      0000000., 1410.85,      0000000.,   47.19,   1050.),
                ( 19700000,      0000000., 1410.85,      0000000.,   47.19,   1100.),
                (  2100000,      0000000., 1410.85,      0000000.,   47.19,   1400.),
                (  2100000,      0000000., 1410.85,      0000000.,   47.19,   5000.)))

    mdb.models['Model-1'].materials['AISI316L-parent'].plastic.CyclicHardening(
        temperatureDependency=ON,parameters=ON, table=(
                (125600000, 153.4,   6.9,   20.),
                ( 97600000, 154.7,   6.9,  275.),
                ( 90900000, 150.6,   6.9,  550.),
                ( 71400000,  57.9,   6.9,  750.),
                ( 66200000,    5.,   6.9,  900.),
                ( 31800000,    5.,   6.9, 1000.),
                ( 25000000,    0.,   6.9, 1050.),
                ( 19700000,    0.,   6.9, 1100.),
                (  2100000,    0.,   6.9, 1400.),
                (  2100000,    0.,   6.9, 5000.)))
    mdb.models['Model-1'].materials['AISI316L-parent'].plastic.AnnealTemperature(table=(
        (1050.0, ), ))

    mdb.models['Model-1'].materials['AISI316L-parent'].SpecificHeat(
        temperatureDependency=ON, table=((492, 20.0), (502, 
        100.0), (514, 200.0), (526, 300.0), (538, 
        400.0), (550, 500.0), (562, 600.0), (575, 
        700.0), (587, 800.0), (599, 900.0), (611,
        1000.0), (623, 1100.0), (635, 1200), (647, 1300.0), (659, 1400.0)))
    mdb.models['Model-1'].materials['AISI316L-parent'].Conductivity(
        temperatureDependency=ON, table=((14.12, 20.0), (15.26, 
        100.0), (16.69, 200.0), (18.11, 300.0), (19.54, 
        400.0), (20.96, 500.0), (22.38, 600.0), (23.81, 
        700.0), (25.23, 800.0), (26.66, 900.0), (28.08, 
        1000.0), (29.50, 1100.0), (30.93, 1200.0), (32.35, 1300.0), (33.78, 1400.0)))
    mdb.models['Model-1'].materials['AISI316L-parent'].Expansion(table=((1.456e-05, 20.0), 
        (1.539e-05, 100.0), (1.621e-05, 200.0), (1.686e-05, 300.0), (
        1.737e-05, 400.0), (1.778e-05, 500.0), (1.812e-05, 600.0), (
        1.843e-05, 700.0), (1.872e-05, 800.0), (1.899e-05, 900.0), (
        1.927e-05, 1000.0), (1.953e-05, 1100.0), (1.979e-05, 1200.0), (
        2.002e-05, 1300.0), (2.021e-05, 1400.0)), 
        temperatureDependency=ON)  
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
        material='AISI316L-parent', thickness=None)
    c = p.cells
    region = regionToolset.Region(cells=c)
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

    #Step
    #decrease the time incr
    mdb.models['Model-1'].StaticStep(name='Welding', 
        previous='Initial', timePeriod= bead_length/arc_speed + 
        stop_time_at_start + stop_time_at_end, maxNumInc=1000000000, 
        timeIncrementationMethod=FIXED, initialInc=0.1, nlgeom=ON)
    mdb.models['Model-1'].StaticStep(name='Cooling_1', 
        previous='Welding', timePeriod=70, initialInc=0.1,
        minInc=0.05, maxInc=5.0, maxNumInc=100000,nlgeom=ON)
    mdb.models['Model-1'].StaticStep(name='Cooling_2', 
        previous='Cooling_1', timePeriod=1000, initialInc=5,
        minInc=1, maxInc=50, maxNumInc=100000,nlgeom=ON)

    # Assembly
    a = mdb.models['Model-1'].rootAssembly
    myinstance = a.Instance(name='Part-assembly', part=p, dependent=ON)
    c1 = a.instances['Part-assembly'].cells
    f1 = a.instances['Part-assembly'].faces
    e1 = a.instances['Part-assembly'].edges
    v1 = a.instances['Part-assembly'].vertices
    n1 = a.instances['Part-assembly'].nodes

    # Attribute
    mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-8)

    # Boundary Condition
    left_vertice = v1.findAt(((specimen_length/2,0,0),))
    right_vertice = v1.findAt(((-specimen_length/2,0,0),))
    region = a.Set(vertices=left_vertice, name='left_vertice')
    mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
        region=region, u1=SET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    region = a.Set(vertices=right_vertice, name='right_vertice')
    mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial', 
        region=region, u1=UNSET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    front_face_1_node = (bead_length/2+finest_mesh_size, 0, finest_mesh_size)
    front_face_1 = f1.findAt((front_face_1_node,))[0]
    tar_face = front_face_1.getFacesByFaceAngle(20)
    region = regionToolset.Region(faces=tar_face)
    mdb.models['Model-1'].YsymmBC(name='BC-3', createStepName='Initial', 
        region=region, localCsys=None)

    # Predefined field
    region = regionToolset.Region(vertices=v1, edges=e1, faces=f1, 
            cells=c1)
    mdb.models['Model-1'].Temperature(name='Predefined Field-1', 
        createStepName='Initial', region=region, distributionType=UNIFORM, 
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(room_temperature, 
        ))
    mdb.models['Model-1'].Temperature(name='Predefined Field-2', 
        createStepName='Welding', distributionType=FROM_FILE, 
        fileName='Thermal_field_odb_root', 
        beginStep=1, beginIncrement=1, endStep=1, endIncrement=0, 
        interpolate=OFF, absoluteExteriorTolerance=0.0, exteriorTolerance=0.05)
    mdb.models['Model-1'].Temperature(name='Predefined Field-3', 
        createStepName='Cooling_1', distributionType=FROM_FILE, 
        fileName='Thermal_field_odb_root', 
        beginStep=2, beginIncrement=1, endStep=2, endIncrement=0, 
        interpolate=OFF, absoluteExteriorTolerance=0.0, exteriorTolerance=0.05)
    mdb.models['Model-1'].Temperature(name='Predefined Field-4', 
        createStepName='Cooling_2', distributionType=FROM_FILE, 
        fileName='Thermal_field_odb_root', 
        beginStep=3, beginIncrement=1, endStep=3, endIncrement=0, 
        interpolate=OFF, absoluteExteriorTolerance=0.0, exteriorTolerance=0.05)

    # Mesh
    p = mdb.models['Model-1'].parts['Part-1']
    e = p.edges
    p.seedPart(size=finest_mesh_size*4, deviationFactor=0.1, minSizeFactor=0.1)
    pickedEdges = (e.findAt(((0,0,0),))) 
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size*2, deviationFactor=0.1, minSizeFactor=0.1,
        constraint=FINER)
    pickedEdges = (e.findAt(((HAZ_length/2, finest_mesh_size ,0),)))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size*2, deviationFactor=0.1, minSizeFactor=0.1,
        constraint=FINER)
    pickedEdges = (e.findAt(((HAZ_length/2, 0,finest_mesh_size),)))
    p.seedEdgeBySize(edges=pickedEdges, size=finest_mesh_size, deviationFactor=0.1, minSizeFactor=0.1,
        constraint=FINER)

    p.generateMesh()    
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, 
        secondOrderAccuracy=OFF, distortionControl=DEFAULT)
    p = mdb.models['Model-1'].parts['Part-1']
    c = p.cells
    p.setElementType(regions=(c, ), elemTypes=(elemType1,))

    # Output
    mdb.models['Model-1'].FieldOutputRequest(name='F-Output-1', 
        createStepName='Welding', variables=('S', 'U', 'NT'))
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()

    # create inp
    name = 'M-'+ str(int(specimen_length*1000)) +'-'+ str(int(specimen_width*1000)) + '-' + str(
        int(specimen_hight*1000)) + '-' + str(int(bead_length*1000)) + '-'+ str(
        int(arc_speed*1000000)) + '-' + str(int(heat_input)) 
    mdb.Job(name=name, model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs[name].writeInput(consistencyChecking=OFF)

def create_arg_list(arg):
    # print(arg)
    l_limit = float(arg.split('-')[-3])
    u_limit = float(arg.split('-')[-2])
    num = int(arg.split('-')[-1])
    para_list = np.linspace(l_limit, u_limit, num)
    return para_list

#init parameter dictionary
para_dic = OrderedDict()
para_dic['specimen_length'] = []
para_dic['specimen_width'] = []
para_dic['specimen_hight'] = []
para_dic['bead_length'] = []
para_dic['arc_speed'] = []
para_dic['heat_input'] = []
sys_argv = sys.argv[-6:]
for index, item in enumerate(para_dic):
    if '-' in sys_argv[index]:
        para_list = create_arg_list(sys_argv[index])
        para_dic[item] = para_list
    else:
        para_dic[item] = [float(sys_argv[index])]

# os.chdir(r'Abaqus_scripts/Thermal_inp_files')

# Creat Abaqus Thermal analysis input files
for specimen_length in para_dic['specimen_length']:
    for specimen_width in para_dic['specimen_width']:
        for specimen_hight in para_dic['specimen_hight']:
            for bead_length in para_dic['bead_length']:
                for arc_speed in para_dic['arc_speed']:
                    for heat_input in para_dic['heat_input']:
                        # Set working directory
                        os.chdir(r'./Thermal_inp_files')
                        create_thermal_analysis_model(specimen_length,specimen_width,specimen_hight,bead_length,arc_speed,heat_input)
                        # Set working directory
                        os.chdir(r'../Mechanical_inp_files') 
                        create_Mechanical_analysis_model(specimen_length,specimen_width,specimen_hight,bead_length,arc_speed,heat_input)
                        os.chdir(r'../') 
                     
                        


