#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:08:18 2021

@author: alessandro
"""
#%% The code template
temp='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">

<define>
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
  <matrix coldim="2" name="cameraBodySurf_REFLECTIVITY" values="5e-08 0 0.00015 0"/>
  <matrix coldim="2" name="cameraBodySurf_EFFICIENCY" values="5e-08 1 0.00015 1"/>
</define>

<materials>
  <isotope N="34" Z="29" name="Cu63">
    <atom unit="g/mole" value="62.9295975"/>
  </isotope>
  <isotope N="36" Z="29" name="Cu65">
    <atom unit="g/mole" value="64.9277895"/>
  </isotope>
  <element name="Cu_element">
    <fraction ref="Cu63" n="0.6917" />
    <fraction ref="Cu65" n="0.3083" />
  </element>

  <isotope N="28" Z="26" name="Fe54">
    <atom unit="g/mole" value="53.939609"/>
  </isotope>
  <isotope N="30" Z="26" name="Fe56">
    <atom unit="g/mole" value="55.9349363"/>
  </isotope>
  <isotope N="31" Z="26" name="Fe57">
    <atom unit="g/mole" value="56.9353928"/>
  </isotope>
  <isotope N="32" Z="26" name="Fe58">
    <atom unit="g/mole" value="57.9332744"/>
  </isotope>
  <element name="Fe_element">
    <fraction ref="Fe54" n="0.0585" />
    <fraction ref="Fe56" n="0.9175" />
    <fraction ref="Fe57" n="0.0212" />
    <fraction ref="Fe58" n="0.0028" />
  </element>

  <isotope N="14" Z="14" name="Si28">
    <atom unit="g/mole" value="27.9769265350"/>
  </isotope>
  <isotope N="15" Z="14" name="Si29">
    <atom unit="g/mole" value="28.9764946653"/>
  </isotope>
  <isotope N="16" Z="14" name="Si30">
    <atom unit="g/mole" value="29.973770137"/>
  </isotope>
  <element name="Si_element">
    <fraction ref="Si28" n="0.922" />
    <fraction ref="Si29" n="0.047" />
    <fraction ref="Si30" n="0.031" />
  </element>

  <isotope N="30" Z="25" name="Mn55">
    <atom unit="g/mole" value="54.9380451"/>
  </isotope>
  <element name="Mn_element">
    <fraction ref="Mn55" n="1" />
  </element>

  <isotope N="12" Z="12" name="Mg24">
    <atom unit="g/mole" value="23.985041697"/>
  </isotope>
  <isotope N="13" Z="12" name="Mg25">
    <atom unit="g/mole" value="24.98583696"/>
  </isotope>
  <isotope N="14" Z="12" name="Mg26">
    <atom unit="g/mole" value="25.98259297"/>
  </isotope>
  <element name="Mg_element">
    <fraction ref="Mg24" n="0.79" />
    <fraction ref="Mg25" n="0.1" />
    <fraction ref="Mg26" n="0.11" />
  </element>

  <isotope N="34" Z="30" name="Zn64">
    <atom unit="g/mole" value="63.9291422"/>
  </isotope>
  <isotope N="36" Z="30" name="Zn66">
    <atom unit="g/mole" value="65.9260334"/>
  </isotope>
  <isotope N="37" Z="30" name="Zn67">
    <atom unit="g/mole" value="66.9271273"/>
  </isotope>
  <isotope N="38" Z="30" name="Zn68">
    <atom unit="g/mole" value="67.9248442"/>
  </isotope>
  <isotope N="40" Z="30" name="Zn70">
    <atom unit="g/mole" value="69.9253193"/>
  </isotope>
  <element name="Zn_element">
    <fraction ref="Zn64" n="0.492" />
    <fraction ref="Zn66" n="0.277" />
    <fraction ref="Zn67" n="0.040" />
    <fraction ref="Zn68" n="0.185" />
    <fraction ref="Zn70" n="0.006" />
  </element>

  <isotope N="26" Z="24" name="Cr50">
    <atom unit="g/mole" value="49.9460442"/>
  </isotope>
  <isotope N="28" Z="24" name="Cr52">
    <atom unit="g/mole" value="51.9405075"/>
  </isotope>
  <isotope N="29" Z="24" name="Cr53">
    <atom unit="g/mole" value="52.9406494"/>
  </isotope>
  <isotope N="30" Z="24" name="Cr54">
    <atom unit="g/mole" value="53.9388804"/>
  </isotope>
  <element name="Cr_element">
    <fraction ref="Cr50" n="0.04345" />
    <fraction ref="Cr52" n="0.83789" />
    <fraction ref="Cr53" n="0.09501" />
    <fraction ref="Cr54" n="0.02365" />
  </element>

  <isotope N="122" Z="81" name="Ti203">
    <atom unit="g/mole" value="202.9723442"/>
  </isotope>
  <isotope N="124" Z="81" name="Ti205">
    <atom unit="g/mole" value="204.9744275"/>
  </isotope>
  <element name="Ti_element">
    <fraction ref="Ti203" n="0.295" />
    <fraction ref="Ti205" n="0.705" />
  </element>

  <isotope N="14" Z="13" name="Al27">
    <atom unit="g/mole" value="26.98153841"/>
  </isotope>
  <element name="Al_element">
    <fraction ref="Al27" n="1" />
  </element>

  <material name="G4_Al6082" state="solid">
    <!-- <property name="RINDEX" ref="Al6082_RINDEX"/> -->
    <!-- <property name="GROUPVEL" ref="Al6082_GROUPVEL"/> -->
    <!-- <T unit="K" value="293.15"/> -->
    <!-- <MEE unit="eV" value="85.7"/> -->
    <D unit="g/cm3" value="2.96"/>
    <fraction n="0.9715" ref="Al_element"/>
    <fraction n="0.001" ref="Cu_element"/>
    <fraction n="0.005" ref="Fe_element"/>
    <fraction n="0.004" ref="Mn_element"/>
    <fraction n="0.006" ref="Mg_element"/>
    <fraction n="0.007" ref="Si_element"/>
    <fraction n="0.002" ref="Zn_element"/>
    <fraction n="0.0025" ref="Cr_element"/>
    <fraction n="0.001" ref="Ti_element"/>
  </material>
</materials>

<solids>
  <box lunit="mm" name="cameraBodyExt_solid" x="{0:.03f} + {1:.03f}" y="{0:.03f} + {1:.03f}" z="{2:.03f}"/>
  <box lunit="mm" name="cameraBodyInt_solid" x="{0:.03f}" y="{0:.03f}" z="{2:.03f} - {1:.03f}"/>
  <subtraction name="cameraBody_solid">
    <first ref="cameraBodyExt_solid"/>
    <second ref="cameraBodyInt_solid"/>
    <position name="cameraBody_pos" unit="mm" x="{3:.03f}" y="{4:.03f}" z="{1:.03f}/2"/>
  </subtraction>
  <opticalsurface finish="0" model="0" name="cameraBodySurf" type="1" value="1">
    <property name="REFLECTIVITY" ref="cameraBodySurf_REFLECTIVITY"/>
    <property name="EFFICIENCY" ref="cameraBodySurf_EFFICIENCY"/>
  </opticalsurface>
</solids>

<structure>
  <volume name="cameraBody">
    <materialref ref="G4_Al6082"/>
    <solidref ref="cameraBody_solid"/>
  </volume>
  <skinsurface name="cameraBody_Surface" surfaceproperty="cameraBodySurf">
    <volumeref ref="cameraBody"/>
  </skinsurface>
</structure>



<setup name="Default" version="1.0">
  <world ref="cameraBody"/>
</setup>

</gdml>
'''
#%% Making of the body file
def make_body(mask_side,body_thick,assembly_thick,body_x,body_y,path):
    with open('{0}/cameraBody.gdml'.format(path), 'w') as f:
        print(temp.format(mask_side,body_thick,assembly_thick,body_x,body_y),file=f)
        
        
        
        
        
        
        