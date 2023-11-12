#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 18:21:29 2021

@author: alessandro
"""
#%% Including the ancillary files
from make_mask import make_mask
from make_body import make_body
from make_volume import make_volume
from make_det import make_det
from make_cold_dem import make_cold_dem
import mask_generation as mg
import os
import argparse as ap
#%% Main function code
if  __name__ == "__main__":    
    parser = ap.ArgumentParser(description="Generation of the .gdml files")
    group1=parser.add_argument_group('Initial settings')
    group1.add_argument("--folder",help="select the folder to save the .gdmls in",default='./gdmls')
    group1.add_argument("--mask_type",help="select the type of mask",choices=['MURA', 'random', 'odd_dim','even_dim'],default='even_dim')
    group1.add_argument("--mask_rank",help="select the rank of the base MURA",type=int,default=13)
    group2=parser.add_argument_group('Mask parameters')
    group2.add_argument('--b',type=float,help="Mask-detector distance",default=20)
    group2.add_argument('--mask_thick',type=float,help="Mask thickness",default=0.1)
    group2.add_argument('--dim',type=float,help="half-plane size",default=50.)
    group2.add_argument('--hole_pitch',type=float,help="pitch between holes",default=3.2)
    group2.add_argument('--mask_hole',type=float,help="size of the holes",default=3.)
    group2.add_argument('--hole_z',type=float,help="thickness of the holes",default=0.2)
    group3=parser.add_argument_group('Detector parameters')
    group3.add_argument('--det_thick',type=float,help="Detector thickness",default=1.)
    group3.add_argument('--det_side',type=float,help="Detector side dimension",default=51.2)
    group3.add_argument('--det_cellcount',type=int,help="number of cells per side",default=16)
    group3.add_argument('--det_cell_border', type=float,help='Detector cell border',default=0.2)
    group3.add_argument('--body_thick',type=int,help="Cam. body thickess",default=1.)
    group4=parser.add_argument_group('Element coordinates')
    group4.add_argument('--body_x',type=float,help="Cam. body x coord.",default=0.)
    group4.add_argument('--body_y',type=float,help="Cam. body y coord.",default=0.)
    group4.add_argument('--mask_x',type=float,help="Mask x coord.",default=0.)
    group4.add_argument('--mask_y',type=float,help="Mask y coord.",default=0.)
    group4.add_argument('--det_x',type=float,help="Detector x coord.",default=1.6)
    group4.add_argument('--det_y',type=float,help="Detector y coord.",default=1.6)
    group4.add_argument('--cam_x',type=float,help="Camera x coord.",default=0.)
    group4.add_argument('--cam_y',type=float,help="Camera y coord.",default=0.)
    group4.add_argument('--cam_z',type=float,help="Camera z coord.",default=0.)
    group5=parser.add_argument_group('Cold demonstrator parameters')
    group5.add_argument('--side_width',type=float,help="Mask side size",default=120)
    group5.add_argument('--distance',type=float,help="Distance between masks",default=500)
    #parser.print_help()
    args=parser.parse_args()
    folder=args.folder
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    masks={'even_dim':mg.gen_lecce_mosaic,'odd_dim':mg.gen_mosaic1,'random':mg.gen_random,'MURA':mg.gen_mura}
    myfunc=args.mask_type
    if myfunc in masks:
        matrix=masks[myfunc](args.mask_rank)
#Parameters: all in mm!
    #mask-sensor distance
    b=args.b
    #thickness of the mask
    mask_thick=args.mask_thick
    #size of the half plane
    dim=args.dim
    #side of the whole mask
    mask_side=dim*2
    #size of the hole (no border)
    mask_hole=args.mask_hole
    #thickness of the hole
    hole_z=args.hole_z
    #pitch for each hole (to be passed to make_mask)
    hole_pitch=args.hole_pitch
    #thickness of the SiPM
    det_thick=args.det_thick
    det_side=args.det_side
#SiPM cell properties
    det_cellcount=args.det_cellcount
    det_celledge=args.det_cell_border
    det_cellsize=det_side/det_cellcount-det_celledge
    body_thick=args.body_thick
    assembly_thick=b + mask_thick / 2 + det_thick / 2 + body_thick
#Coordinates: all in mm!
    body_x=args.body_x
    body_y=args.body_y
    mask_x=args.mask_x
    mask_y=args.mask_y
    det_x=args.det_x
    det_y=args.det_y
    cam_x=args.cam_x
    cam_y=args.cam_y
    cam_z=args.cam_z
#Cold demonstrator parameters
    side_width=args.side_width
    distance=args.distance
#Functions
#calling the make_mask function to make the mask gdml
    make_mask(matrix,mask_thick,hole_pitch,mask_hole,dim,hole_z,folder)
#calling the make_body function to make the body gdml
    make_body(mask_side,body_thick,assembly_thick,body_x,body_y,folder)
#calling the make_volume function to make the volume xml
    make_volume(mask_x,mask_y,det_x,det_y,cam_x,cam_y,cam_z,assembly_thick,mask_thick,det_thick,body_thick,folder)
#calling the make_det function to make the detector gdml
    make_det(det_side, det_thick, det_cellcount, det_cellsize, det_celledge,folder)
#calling the make_cold_demonstrator function to make the c.d. gdml
    make_cold_dem(mask_side,assembly_thick,side_width,distance,folder)
#%%generation of the main.gdml file
    temp1='''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE gdml [
<!ENTITY properties SYSTEM "./properties.xml">
]>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">

<define>
  &properties;
</define>

<materials>
  <isotope N="12" Z="6" name="C12">
    <atom unit="g/mole" value="12"/>
  </isotope>
  <isotope N="13" Z="6" name="C13">
    <atom unit="g/mole" value="13.0034"/>
  </isotope>
  <element name="C_element">
    <fraction ref="C12" n="0.9893" />
    <fraction ref="C13" n="0.0107" />
  </element>

  <isotope N="14" Z="7" name="N14">
    <atom unit="g/mole" value="14.0031"/>
  </isotope>
  <isotope N="15" Z="7" name="N15">
    <atom unit="g/mole" value="15.0001"/>
  </isotope>
  <element name="N_element">
    <fraction n="0.99632" ref="N14"/>
    <fraction n="0.00368" ref="N15"/>
  </element>

  <isotope N="16" Z="8" name="O16">
    <atom unit="g/mole" value="15.9949"/>
  </isotope>
  <isotope N="17" Z="8" name="O17">
    <atom unit="g/mole" value="16.9991"/>
  </isotope>
  <isotope N="18" Z="8" name="O18">
    <atom unit="g/mole" value="17.9992"/>
  </isotope>
  <element name="O_element">
    <fraction n="0.99757" ref="O16"/>
    <fraction n="0.00038" ref="O17"/>
    <fraction n="0.00205" ref="O18"/>
  </element>

  <isotope N="36" Z="18" name="Ar36">
    <atom unit="g/mole" value="35.9675"/>
  </isotope>
  <isotope N="38" Z="18" name="Ar38">
    <atom unit="g/mole" value="37.9627"/>
  </isotope>
  <isotope N="40" Z="18" name="Ar40">
    <atom unit="g/mole" value="39.9624"/>
  </isotope>
  <element name="Ar_element">
    <fraction n="0.003365" ref="Ar36"/>
    <fraction n="0.000632" ref="Ar38"/>
    <fraction n="0.996003" ref="Ar40"/>
  </element>

  <material name="G4_AIR" state="gas">
    <property name="RINDEX" ref="air_RINDEX"/>
    <property name="GROUPVEL" ref="air_GROUPVEL"/>
    <T unit="K" value="293.15"/>
    <MEE unit="eV" value="85.7"/>
    <D unit="g/cm3" value="0.00120479"/>
    <fraction n="0.000124000124000124" ref="C_element"/>
    <fraction n="0.755267755267755" ref="N_element"/>
    <fraction n="0.231781231781232" ref="O_element"/>
    <fraction n="0.0128270128270128" ref="Ar_element"/>
  </material>
</materials>

<solids>
  <box lunit="mm" name="World_solid" x="3500" y="3500" z="3500"/>
</solids>

<structure>
  <volume name="World">
    <materialref ref="G4_AIR"/>
    <solidref ref="World_solid"/>
    <physvol name="vessel_ext_physical">
      <file name="./cold_demonstrator.gdml"/>
      <!-- <file name="./meniscus.gdml"/> -->
      <!-- <file name="./meniscus_curved.gdml"/> -->
      <!-- <rotation name="Rotation_0" unit="deg" x="0" y="-90" z="0"/> to be used only with the curved geometry -->
    </physvol>
  </volume>
</structure>

<setup name="Default" version="1.0">
  <world ref="World"/>
</setup>

</gdml>
'''
    with open('./{0}/main.gdml'.format(folder), 'w') as f:
        print(temp1,file=f)

#%% Generation of the properties.xml file
    temp2='''<matrix coldim="2" name="air_RINDEX" values=" 2e-06 1 
                                                2.03e-06 1 
                                                2.06e-06 1 
                                                2.09e-06 1 
                                                2.12e-06 1 
                                                2.15e-06 1 
                                                2.18e-06 1 
                                                2.21e-06 1 
                                                2.24e-06 1 
                                                2.27e-06 1 
                                                2.3e-06 1 
                                                2.33e-06 1 
                                                2.36e-06 1 
                                                2.39e-06 1 
                                                2.42e-06 1 
                                                2.45e-06 1 
                                                2.48e-06 1 
                                                2.51e-06 1 
                                                2.54e-06 1 
                                                2.57e-06 1 
                                                2.6e-06 1 
                                                2.63e-06 1 
                                                2.66e-06 1 
                                                2.69e-06 1 
                                                2.72e-06 1 
                                                2.75e-06 1 
                                                2.78e-06 1 
                                                2.81e-06 1 
                                                2.84e-06 1 
                                                2.87e-06 1 
                                                2.9e-06 1 
                                                2.93e-06 1 
                                                2.96e-06 1 
                                                2.99e-06 1 
                                                3.02e-06 1 
                                                3.05e-06 1 
                                                3.08e-06 1 
                                                3.11e-06 1 
                                                3.14e-06 1 
                                                3.17e-06 1 
                                                3.2e-06 1 
                                                3.23e-06 1 
                                                3.26e-06 1 
                                                3.29e-06 1 
                                                3.32e-06 1 
                                                3.35e-06 1 
                                                3.38e-06 1 
                                                3.41e-06 1 
                                                3.44e-06 1 
                                                3.47e-06 1"
                                                />
<matrix coldim="2" name="air_GROUPVEL" values=" 2e-06 299.792 
                                                  2.015e-06 299.792 
                                                  2.045e-06 299.792 
                                                  2.075e-06 299.792 
                                                  2.105e-06 299.792 
                                                  2.135e-06 299.792 
                                                  2.165e-06 299.792 
                                                  2.195e-06 299.792 
                                                  2.225e-06 299.792 
                                                  2.255e-06 299.792 
                                                  2.285e-06 299.792 
                                                  2.315e-06 299.792 
                                                  2.345e-06 299.792 
                                                  2.375e-06 299.792 
                                                  2.405e-06 299.792 
                                                  2.435e-06 299.792 
                                                  2.465e-06 299.792 
                                                  2.495e-06 299.792 
                                                  2.525e-06 299.792 
                                                  2.555e-06 299.792 
                                                  2.585e-06 299.792 
                                                  2.615e-06 299.792 
                                                  2.645e-06 299.792 
                                                  2.675e-06 299.792 
                                                  2.705e-06 299.792 
                                                  2.735e-06 299.792 
                                                  2.765e-06 299.792 
                                                  2.795e-06 299.792 
                                                  2.825e-06 299.792 
                                                  2.855e-06 299.792 
                                                  2.885e-06 299.792 
                                                  2.915e-06 299.792 
                                                  2.945e-06 299.792 
                                                  2.975e-06 299.792 
                                                  3.005e-06 299.792 
                                                  3.035e-06 299.792 
                                                  3.065e-06 299.792 
                                                  3.095e-06 299.792 
                                                  3.125e-06 299.792 
                                                  3.155e-06 299.792 
                                                  3.185e-06 299.792 
                                                  3.215e-06 299.792 
                                                  3.245e-06 299.792 
                                                  3.275e-06 299.792 
                                                  3.305e-06 299.792 
                                                  3.335e-06 299.792 
                                                  3.365e-06 299.792 
                                                  3.395e-06 299.792 
                                                  3.425e-06 299.792 
                                                  3.47e-06 299.792"
                                                  />
<matrix coldim="2" name="LAr_RINDEX" values=" 4.1e-06 1.24
                                                4.29e-06 1.24
                                                4.42e-06 1.24
                                                4.57e-06 1.24
                                                4.93e-06 1.25
                                                5.04e-06 1.25
                                                5.28e-06 1.25
                                                5.53e-06 1.26
                                                5.74e-06 1.26
                                                5.95e-06 1.26
                                                6.23e-06 1.27
                                                6.57e-06 1.28
                                                7.05e-06 1.29
                                                7.48e-06 1.3
                                                7.62e-06 1.3
                                                7.82e-06 1.31
                                                8.02e-06 1.32
                                                8.09e-06 1.32
                                                8.2e-06 1.32
                                                8.45e-06 1.34
                                                8.57e-06 1.34
                                                8.66e-06 1.35
                                                8.81e-06 1.36
                                                9.03e-06 1.38
                                                9.24e-06 1.4
                                                9.41e-06 1.42
                                                9.56e-06 1.44
                                                9.77e-06 1.47
                                                1.004e-05 1.52
                                                1.018e-05 1.55
                                                1.033e-05 1.59
                                                1.051e-05 1.64
                                                1.061e-05 1.67"
                                                />
<matrix coldim="2" name="LAr_GROUPVEL" values=" 4.1e-06 241.768
                                                  4.195e-06 241.768
                                                  4.355e-06 241.768
                                                  4.495e-06 241.768
                                                  4.75e-06 217.733
                                                  4.985e-06 239.834
                                                  5.16e-06 239.834
                                                  5.405e-06 203.779
                                                  5.635e-06 237.931
                                                  5.845e-06 237.931
                                                  6.09e-06 202.226
                                                  6.4e-06 204.889
                                                  6.81e-06 210.113
                                                  7.265e-06 204.79
                                                  7.55e-06 230.61
                                                  7.72e-06 177.289
                                                  7.92e-06 175.217
                                                  8.055e-06 227.115
                                                  8.145e-06 227.115
                                                  8.325e-06 150.2
                                                  8.51e-06 223.726
                                                  8.615e-06 130.219
                                                  8.735e-06 154.746
                                                  8.92e-06 137.465
                                                  9.135e-06 132.654
                                                  9.325e-06 119.581
                                                  9.485e-06 111.255
                                                  9.665e-06 105.722
                                                  9.905e-06 90.0509
                                                  1.011e-05 80.9945
                                                  1.0255e-05 69.6444
                                                  1.042e-05 66.4821
                                                  1.061e-05 61.9665"
                                                  />
<matrix coldim="2" name="LAr_RAYLEIGH" values=" 4.1e-06 0
                                                  4.29e-06 900
                                                  4.42e-06 900
                                                  4.57e-06 900
                                                  4.93e-06 900
                                                  5.04e-06 900
                                                  5.28e-06 900
                                                  5.53e-06 900
                                                  5.74e-06 900
                                                  5.95e-06 900
                                                  6.23e-06 900
                                                  6.57e-06 900
                                                  7.05e-06 900
                                                  7.48e-06 900
                                                  7.62e-06 900
                                                  7.82e-06 900
                                                  8.02e-06 900
                                                  8.09e-06 900
                                                  8.2e-06 900
                                                  8.45e-06 900
                                                  8.57e-06 900
                                                  8.66e-06 900
                                                  8.81e-06 900
                                                  9.03e-06 900
                                                  9.24e-06 900
                                                  9.41e-06 900
                                                  9.56e-06 900
                                                  9.77e-06 900
                                                  1.004e-05 900
                                                  1.018e-05 900
                                                  1.033e-05 900
                                                  1.051e-05 900
                                                  1.061e-05 900"
                                                  />
<matrix coldim="2" name="LAr_ABSLENGTH" values="4.1e-06 0
                                                  4.29e-06 5100
                                                  4.42e-06 5100
                                                  4.57e-06 5100
                                                  4.93e-06 5100
                                                  5.04e-06 5100
                                                  5.28e-06 5100
                                                  5.53e-06 5100
                                                  5.74e-06 5100
                                                  5.95e-06 5100
                                                  6.23e-06 5100
                                                  6.57e-06 5100
                                                  7.05e-06 5100
                                                  7.48e-06 5100
                                                  7.62e-06 5100
                                                  7.82e-06 5100
                                                  8.02e-06 5100
                                                  8.09e-06 5100
                                                  8.2e-06 5100
                                                  8.45e-06 5100
                                                  8.57e-06 5100
                                                  8.66e-06 5100
                                                  8.81e-06 5100
                                                  9.03e-06 5100
                                                  9.24e-06 5100
                                                  9.41e-06 5100
                                                  9.56e-06 5100
                                                  9.77e-06 5100
                                                  1.004e-05 5100
                                                  1.018e-05 5100
                                                  1.033e-05 5100
                                                  1.051e-05 5100
                                                  1.061e-05 5100"
                                                  />
<matrix coldim="2" name="LAr_FASTCOMPONENT" values="4.1e-06 3.56e-05
                                                      4.29e-06 9.29e-05
                                                      4.42e-06 0.000141
                                                      4.57e-06 0.000176
                                                      4.93e-06 6.27e-05
                                                      5.04e-06 8.02e-05
                                                      5.28e-06 4.03e-05
                                                      5.53e-06 4.45e-05
                                                      5.74e-06 8.22e-05
                                                      5.95e-06 0.000148
                                                      6.23e-06 0.000225
                                                      6.57e-06 0.000225
                                                      7.05e-06 0.000159
                                                      7.48e-06 7.27e-05
                                                      7.62e-06 0.000131
                                                      7.82e-06 0.000236
                                                      8.02e-06 0.000152
                                                      8.09e-06 0.000199
                                                      8.2e-06 0.000317
                                                      8.45e-06 0.00117
                                                      8.57e-06 0.00244
                                                      8.66e-06 0.00522
                                                      8.81e-06 0.0133
                                                      9.03e-06 0.0538
                                                      9.24e-06 0.158
                                                      9.41e-06 0.413
                                                      9.56e-06 0.745
                                                      9.77e-06 1
                                                      1.004e-05 0.659
                                                      1.018e-05 0.293
                                                      1.033e-05 0.155
                                                      1.051e-05 0.107
                                                      1.061e-05 0.0442"
                                                      />
<matrix coldim="2" name="LAr_SLOWCOMPONENT" values="4.1e-06 3.56e-05
                                                      4.29e-06 9.29e-05
                                                      4.42e-06 0.000141
                                                      4.57e-06 0.000176
                                                      4.93e-06 6.27e-05
                                                      5.04e-06 8.02e-05
                                                      5.28e-06 4.03e-05
                                                      5.53e-06 4.45e-05
                                                      5.74e-06 8.22e-05
                                                      5.95e-06 0.000148
                                                      6.23e-06 0.000225
                                                      6.57e-06 0.000225
                                                      7.05e-06 0.000159
                                                      7.48e-06 7.27e-05
                                                      7.62e-06 0.000131
                                                      7.82e-06 0.000236
                                                      8.02e-06 0.000152
                                                      8.09e-06 0.000199
                                                      8.2e-06 0.000317
                                                      8.45e-06 0.00117
                                                      8.57e-06 0.00244
                                                      8.66e-06 0.00522
                                                      8.81e-06 0.0133
                                                      9.03e-06 0.0538
                                                      9.24e-06 0.158
                                                      9.41e-06 0.413
                                                      9.56e-06 0.745
                                                      9.77e-06 1
                                                      1.004e-05 0.659
                                                      1.018e-05 0.293
                                                      1.033e-05 0.155
                                                      1.051e-05 0.107
                                                      1.061e-05 0.0442"
                                                      />
<matrix coldim="1" name="LAr_SCINTILLATIONYIELD" values="40000"/>
<matrix coldim="1" name="LAr_RESOLUTIONSCALE" values="2"/>
<matrix coldim="1" name="LAr_FASTTIMECONSTANT" values="7"/>
<matrix coldim="1" name="LAr_SLOWTIMECONSTANT" values="1600"/>
<matrix coldim="1" name="LAr_YIELDRATIO" values="0.8"/>
<matrix coldim="2" name="meniscus_surface_REFLECTIVITY" values=" 0 0 
                                                              10 0"
                                                              />
<matrix coldim="2" name="meniscus_surface_EFFICIENCY" values=" 0 1 
                                                            10 1"
                                                            />
<matrix coldim="2" name="Si_RINDEX" values="2e-06 1.46 
                                              2.03e-06 1.46 
                                              2.06e-06 1.46 
                                              2.09e-06 1.46 
                                              2.12e-06 1.46 
                                              2.15e-06 1.46 
                                              2.18e-06 1.46 
                                              2.21e-06 1.46 
                                              2.24e-06 1.46 
                                              2.27e-06 1.46 
                                              2.3e-06 1.46 
                                              2.33e-06 1.46 
                                              2.36e-06 1.46 
                                              2.39e-06 1.46 
                                              2.42e-06 1.46 
                                              2.45e-06 1.46 
                                              2.48e-06 1.46 
                                              2.51e-06 1.46 
                                              2.54e-06 1.46 
                                              2.57e-06 1.46 
                                              2.6e-06 1.46 
                                              2.63e-06 1.46 
                                              2.66e-06 1.46 
                                              2.69e-06 1.46 
                                              2.72e-06 1.46 
                                              2.75e-06 1.46 
                                              2.78e-06 1.46 
                                              2.81e-06 1.46 
                                              2.84e-06 1.46 
                                              2.87e-06 1.46 
                                              2.9e-06 1.46 
                                              2.93e-06 1.46 
                                              2.96e-06 1.46 
                                              2.99e-06 1.46 
                                              3.02e-06 1.46 
                                              3.05e-06 1.46 
                                              3.08e-06 1.46 
                                              3.11e-06 1.46 
                                              3.14e-06 1.46 
                                              3.17e-06 1.46 
                                              3.2e-06 1.46 
                                              3.23e-06 1.46 
                                              3.26e-06 1.46 
                                              3.29e-06 1.46 
                                              3.32e-06 1.46 
                                              3.35e-06 1.46 
                                              3.38e-06 1.46 
                                              3.41e-06 1.46 
                                              3.44e-06 1.46 
                                              3.47e-06 1.46"
                                              />
<matrix coldim="2" name="Si_GROUPVEL" values="2e-06 205.337 
                                                2.015e-06 205.337 
                                                2.045e-06 205.337 
                                                2.075e-06 205.337 
                                                2.105e-06 205.337 
                                                2.135e-06 205.337 
                                                2.165e-06 205.337 
                                                2.195e-06 205.337 
                                                2.225e-06 205.337 
                                                2.255e-06 205.337 
                                                2.285e-06 205.337 
                                                2.315e-06 205.337 
                                                2.345e-06 205.337 
                                                2.375e-06 205.337 
                                                2.405e-06 205.337 
                                                2.435e-06 205.337 
                                                2.465e-06 205.337 
                                                2.495e-06 205.337 
                                                2.525e-06 205.337 
                                                2.555e-06 205.337 
                                                2.585e-06 205.337 
                                                2.615e-06 205.337 
                                                2.645e-06 205.337 
                                                2.675e-06 205.337 
                                                2.705e-06 205.337 
                                                2.735e-06 205.337 
                                                2.765e-06 205.337 
                                                2.795e-06 205.337 
                                                2.825e-06 205.337 
                                                2.855e-06 205.337 
                                                2.885e-06 205.337 
                                                2.915e-06 205.337 
                                                2.945e-06 205.337 
                                                2.975e-06 205.337 
                                                3.005e-06 205.337 
                                                3.035e-06 205.337 
                                                3.065e-06 205.337 
                                                3.095e-06 205.337 
                                                3.125e-06 205.337 
                                                3.155e-06 205.337 
                                                3.185e-06 205.337 
                                                3.215e-06 205.337 
                                                3.245e-06 205.337 
                                                3.275e-06 205.337 
                                                3.305e-06 205.337 
                                                3.335e-06 205.337 
                                                3.365e-06 205.337 
                                                3.395e-06 205.337 
                                                3.425e-06 205.337 
                                                3.47e-06 205.337"
                                                />
<matrix coldim="2" name="photoDetectorBodySurf_REFLECTIVITY" values="5e-08 0 0.00015 0"/>
<matrix coldim="2" name="photoDetectorBodySurf_EFFICIENCY" values="5e-08 1 0.00015 1"/>
'''
    with open('./{0}/properties.xml'.format(folder), 'w') as s:
        print(temp2,file=s)
    
    
    