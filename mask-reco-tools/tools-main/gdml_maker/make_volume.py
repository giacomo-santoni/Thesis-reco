#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:17:21 2021

@author: alessandro
"""

#%% the code template:
temp='''
<volume name="cam_volume">
    <materialref ref="G4_lAr"/>
    <solidref ref="cam_solid"/>
    <physvol name="cameraAssembly_mask">
      <file name="./codedApertureMask.gdml"/>
      <rotation name="Rotation_0" unit="deg" x="0" y="0" z="0"/>
      <position name="Position_0" unit="mm" x="{0:.03f}" y="{1:.03f}" z="{7:.03f} / 2 - {8:.03f} / 2"/>
    </physvol>
    <physvol name="cameraAssembly_photoDetector">
      <file name="./photoDetector.gdml"/>
      <position name="Position_1" unit="mm" x="{2:.03f}" y="{3:.03f}" z="- {7:.03f} / 2 + {9:.03f} / 2 + {10:.03f}"/>
    </physvol>
    <physvol name="cameraAssembly_body">
      <file name="./cameraBody.gdml"/>
      <position name="Position_2" unit="mm" x="{4:.03f}" y="{5:.03f}" z="{6:.03f}"/>
    </physvol>
    <auxiliary auxtype="Camera" auxvalue=""/>
  </volume>
'''
#%% Making of the volume file
def make_volume(mask_x,mask_y,det_x,det_y,cam_x,cam_y,cam_z,assembly_thick,mask_thick,det_thick,cam_thick,path):
    with open('{0}/cam_volume.xml'.format(path), 'w') as f:
        print(temp.format(mask_x,mask_y,det_x,det_y,cam_x,cam_y,cam_z,assembly_thick,mask_thick,det_thick,cam_thick),file=f)
        
        
        