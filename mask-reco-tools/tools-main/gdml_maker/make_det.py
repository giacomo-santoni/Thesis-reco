#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:26:27 2021

@author: alessandro
"""
#%% The code template
temp='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<!DOCTYPE gdml [
<!ENTITY properties SYSTEM "./properties.xml">
]>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">

<define>
  &properties;
</define>

<materials>
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

  <material name="Silicon" state="solid">
    <property name="RINDEX" ref="Si_RINDEX"/>
    <property name="GROUPVEL" ref="Si_GROUPVEL"/>
    <T unit="K" value="293.15"/>
    <MEE unit="eV" value="50.129159403018"/>
    <D unit="g/cm3" value="2.3290"/>
    <fraction n="1" ref="Si_element"/>
  </material>
</materials>

<solids>
  <box lunit="mm" name="photoDetector_solid" x="{0:.03f}" y="{0:.03f}" z="{1:.03f}"/>
  <opticalsurface finish="0" model="0" name="opSurfphotoDetector_solid" type="1" value="1">
    <property name="REFLECTIVITY" ref="photoDetectorBodySurf_REFLECTIVITY"/>
    <property name="EFFICIENCY" ref="photoDetectorBodySurf_EFFICIENCY"/>
  </opticalsurface>
</solids>

<structure>
  <volume name="photoDetector">
    <materialref ref="Silicon"/>
    <solidref ref="photoDetector_solid"/>
    <auxiliary auxtype="Sensor" auxvalue="S14160-6050HS">
      <auxiliary auxtype="cellcount" auxvalue="{2}"/>
      <auxiliary auxtype="cellsize" auxunit="mm" auxvalue="{3:.03f}"/>
      <auxiliary auxtype="celledge" auxunit="mm" auxvalue="{4:.03f}"/>
    </auxiliary>
  </volume>
  <skinsurface name="photoDetectorSurface_volume" surfaceproperty="opSurfphotoDetector_solid">
    <volumeref ref="photoDetector"/>
  </skinsurface>
</structure>



<setup name="Default" version="1.0">
  <world ref="photoDetector"/>
</setup>

</gdml>
'''
#%% Making of the photodetector file
def make_det(det_side,det_thick,cellcount,cellsize,celledge,path):
    with open('{0}/photoDetector.gdml'.format(path), 'w') as f:
        print(temp.format(det_side,det_thick,cellcount,cellsize,celledge),file=f)