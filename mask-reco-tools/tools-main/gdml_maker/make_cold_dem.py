#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:21:42 2021

@author: alessandro
"""
#%% the gdml text
# <?xml version="1.0" encoding="UTF-8"?>
# <!DOCTYPE gdml [
# <!ENTITY variables SYSTEM "./variables.xml">
# <!ENTITY properties SYSTEM "./properties.xml">
# <!ENTITY cam_volume SYSTEM "./cam_volume.xml">
# ]>
# <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">

# <define>
#   &variables;
#   &properties; 
# </define>
#%%
# <materials>
#   <isotope N="12" Z="6" name="C12">
#     <atom unit="g/mole" value="12"/>
#   </isotope>
#   <isotope N="13" Z="6" name="C13">
#     <atom unit="g/mole" value="13.0034"/>
#   </isotope>
#   <element name="C_element">
#     <fraction ref="C12" n="0.9893" />
#     <fraction ref="C13" n="0.0107" />
#   </element>

#   <isotope N="14" Z="7" name="N14">
#     <atom unit="g/mole" value="14.0031"/>
#   </isotope>
#   <isotope N="15" Z="7" name="N15">
#     <atom unit="g/mole" value="15.0001"/>
#   </isotope>
#   <element name="N_element">
#     <fraction n="0.99632" ref="N14"/>
#     <fraction n="0.00368" ref="N15"/>
#   </element>

#   <isotope N="16" Z="8" name="O16">
#     <atom unit="g/mole" value="15.9949"/>
#   </isotope>
#   <isotope N="17" Z="8" name="O17">
#     <atom unit="g/mole" value="16.9991"/>
#   </isotope>
#   <isotope N="18" Z="8" name="O18">
#     <atom unit="g/mole" value="17.9992"/>
#   </isotope>
#   <element name="O_element">
#     <fraction n="0.99757" ref="O16"/>
#     <fraction n="0.00038" ref="O17"/>
#     <fraction n="0.00205" ref="O18"/>
#   </element>

#   <isotope N="36" Z="18" name="Ar36">
#     <atom unit="g/mole" value="35.9675"/>
#   </isotope>
#   <isotope N="38" Z="18" name="Ar38">
#     <atom unit="g/mole" value="37.9627"/>
#   </isotope>
#   <isotope N="40" Z="18" name="Ar40">
#     <atom unit="g/mole" value="39.9624"/>
#   </isotope>
#   <element name="Ar_element">
#     <fraction n="0.003365" ref="Ar36"/>
#     <fraction n="0.000632" ref="Ar38"/>
#     <fraction n="0.996003" ref="Ar40"/>
#   </element>

#    <isotope N="34" Z="29" name="Cu63">
#     <atom unit="g/mole" value="62.9295975"/>
#   </isotope>
#   <isotope N="36" Z="29" name="Cu65">
#     <atom unit="g/mole" value="64.9277895"/>
#   </isotope>
#   <element name="Cu_element">
#     <fraction ref="Cu63" n="0.6917" />
#     <fraction ref="Cu65" n="0.3083" />
#   </element>

#   <isotope N="28" Z="26" name="Fe54">
#     <atom unit="g/mole" value="53.939609"/>
#   </isotope>
#   <isotope N="30" Z="26" name="Fe56">
#     <atom unit="g/mole" value="55.9349363"/>
#   </isotope>
#   <isotope N="31" Z="26" name="Fe57">
#     <atom unit="g/mole" value="56.9353928"/>
#   </isotope>
#   <isotope N="32" Z="26" name="Fe58">
#     <atom unit="g/mole" value="57.9332744"/>
#   </isotope>
#   <element name="Fe_element">
#     <fraction ref="Fe54" n="0.0585" />
#     <fraction ref="Fe56" n="0.9175" />
#     <fraction ref="Fe57" n="0.0212" />
#     <fraction ref="Fe58" n="0.0028" />
#   </element>

#   <isotope N="14" Z="14" name="Si28">
#     <atom unit="g/mole" value="27.9769265350"/>
#   </isotope>
#   <isotope N="15" Z="14" name="Si29">
#     <atom unit="g/mole" value="28.9764946653"/>
#   </isotope>
#   <isotope N="16" Z="14" name="Si30">
#     <atom unit="g/mole" value="29.973770137"/>
#   </isotope>
#   <element name="Si_element">
#     <fraction ref="Si28" n="0.922" />
#     <fraction ref="Si29" n="0.047" />
#     <fraction ref="Si30" n="0.031" />
#   </element>

#   <isotope N="30" Z="25" name="Mn55">
#     <atom unit="g/mole" value="54.9380451"/>
#   </isotope>
#   <element name="Mn_element">
#     <fraction ref="Mn55" n="1" />
#   </element>

#   <isotope N="12" Z="12" name="Mg24">
#     <atom unit="g/mole" value="23.985041697"/>
#   </isotope>
#   <isotope N="13" Z="12" name="Mg25">
#     <atom unit="g/mole" value="24.98583696"/>
#   </isotope>
#   <isotope N="14" Z="12" name="Mg26">
#     <atom unit="g/mole" value="25.98259297"/>
#   </isotope>
#   <element name="Mg_element">
#     <fraction ref="Mg24" n="0.79" />
#     <fraction ref="Mg25" n="0.1" />
#     <fraction ref="Mg26" n="0.11" />
#   </element>

#   <isotope N="34" Z="30" name="Zn64">
#     <atom unit="g/mole" value="63.9291422"/>
#   </isotope>
#   <isotope N="36" Z="30" name="Zn66">
#     <atom unit="g/mole" value="65.9260334"/>
#   </isotope>
#   <isotope N="37" Z="30" name="Zn67">
#     <atom unit="g/mole" value="66.9271273"/>
#   </isotope>
#   <isotope N="38" Z="30" name="Zn68">
#     <atom unit="g/mole" value="67.9248442"/>
#   </isotope>
#   <isotope N="40" Z="30" name="Zn70">
#     <atom unit="g/mole" value="69.9253193"/>
#   </isotope>
#   <element name="Zn_element">
#     <fraction ref="Zn64" n="0.492" />
#     <fraction ref="Zn66" n="0.277" />
#     <fraction ref="Zn67" n="0.040" />
#     <fraction ref="Zn68" n="0.185" />
#     <fraction ref="Zn70" n="0.006" />
#   </element>

#   <isotope N="26" Z="24" name="Cr50">
#     <atom unit="g/mole" value="49.9460442"/>
#   </isotope>
#   <isotope N="28" Z="24" name="Cr52">
#     <atom unit="g/mole" value="51.9405075"/>
#   </isotope>
#   <isotope N="29" Z="24" name="Cr53">
#     <atom unit="g/mole" value="52.9406494"/>
#   </isotope>
#   <isotope N="30" Z="24" name="Cr54">
#     <atom unit="g/mole" value="53.9388804"/>
#   </isotope>
#   <element name="Cr_element">
#     <fraction ref="Cr50" n="0.04345" />
#     <fraction ref="Cr52" n="0.83789" />
#     <fraction ref="Cr53" n="0.09501" />
#     <fraction ref="Cr54" n="0.02365" />
#   </element>

#   <isotope N="122" Z="81" name="Ti203">
#     <atom unit="g/mole" value="202.9723442"/>
#   </isotope>
#   <isotope N="124" Z="81" name="Ti205">
#     <atom unit="g/mole" value="204.9744275"/>
#   </isotope>
#   <element name="Ti_element">
#     <fraction ref="Ti203" n="0.295" />
#     <fraction ref="Ti205" n="0.705" />
#   </element>

#   <isotope N="14" Z="13" name="Al27">
#     <atom unit="g/mole" value="26.98153841"/>
#   </isotope>
#   <element name="Al_element">
#     <fraction ref="Al27" n="1" />
#   </element>
#%%
#   <material name="G4_Al6082" state="solid">
#     <!-- <property name="RINDEX" ref="Al6082_RINDEX"/> -->
#     <!-- <property name="GROUPVEL" ref="Al6082_GROUPVEL"/> -->
#     <!-- <T unit="K" value="293.15"/> -->
#     <!-- <MEE unit="eV" value="85.7"/> -->
#     <D unit="g/cm3" value="2.96"/>
#     <fraction n="0.9715" ref="Al_element"/>
#     <fraction n="0.001" ref="Cu_element"/>
#     <fraction n="0.005" ref="Fe_element"/>
#     <fraction n="0.004" ref="Mn_element"/>
#     <fraction n="0.006" ref="Mg_element"/>
#     <fraction n="0.007" ref="Si_element"/>
#     <fraction n="0.002" ref="Zn_element"/>
#     <fraction n="0.0025" ref="Cr_element"/>
#     <fraction n="0.001" ref="Ti_element"/>
#   </material>

#   <material name="G4_AIR" state="gas">
#     <property name="RINDEX" ref="air_RINDEX"/>
#     <property name="GROUPVEL" ref="air_GROUPVEL"/>
#     <T unit="K" value="293.15"/>
#     <MEE unit="eV" value="85.7"/>
#     <D unit="g/cm3" value="0.00120479"/>
#     <fraction n="0.000124000124000124" ref="C_element"/>
#     <fraction n="0.755267755267755" ref="N_element"/>
#     <fraction n="0.231781231781232" ref="O_element"/>
#     <fraction n="0.0128270128270128" ref="Ar_element"/>
#   </material>

#   <material name="G4_lAr" state="liquid">
#     <property name="RINDEX" ref="LAr_RINDEX"/>
#     <property name="GROUPVEL" ref="LAr_GROUPVEL"/>
#     <property name="RAYLEIGH" ref="LAr_RAYLEIGH"/>
#     <property name="ABSLENGTH" ref="LAr_ABSLENGTH"/>
#     <property name="FASTCOMPONENT" ref="LAr_FASTCOMPONENT"/>
#     <property name="SLOWCOMPONENT" ref="LAr_SLOWCOMPONENT"/>
#     <property name="SCINTILLATIONYIELD" ref="LAr_SCINTILLATIONYIELD"/>
#     <property name="RESOLUTIONSCALE" ref="LAr_RESOLUTIONSCALE"/>
#     <property name="FASTTIMECONSTANT" ref="LAr_FASTTIMECONSTANT"/>
#     <property name="SLOWTIMECONSTANT" ref="LAr_SLOWTIMECONSTANT"/>
#     <property name="YIELDRATIO" ref="LAr_YIELDRATIO"/>
#     <T unit="K" value="293.15"/>
#     <MEE unit="eV" value="188"/>
#     <D unit="g/cm3" value="1.396"/>
#     <fraction n="1" ref="Ar_element"/>
#   </material>
# </materials>
#%%
# <solids>
#   <box lunit="mm" name="cam_solid" x="codedApertureMask_side+10" y="codedApertureMask_side+10" z="assembly_thick+10"/>
#per includere tutto il volume della camera uso distance+2*assembly
#le dimensioni laterali sono settate tramite new_par: x e y sono i lati corti
#   <box lunit="mm" name="lar_solid" x="new_par" y="new_par" z="distance+2*assembly_thickness"/>
#   <box lunit="mm" name"air_layer" x="new_par+0.1" y=new_par+0.1" z="distance+2*assembly_thickness+0.1"/>
#   #<box lunit="mm" name="dewar_thickness" x="564" y="564" z="564"/>
#   <box lunit="mm" name="world_small" x="1000" y="1000" z="1000"/>
#   <opticalsurface finish="0" model="0" name="opSurfdewar_solid" type="1" value="1">
#     <property name="REFLECTIVITY" ref="meniscus_surface_REFLECTIVITY"/>
#     <property name="EFFICIENCY" ref="meniscus_surface_EFFICIENCY"/>
#   </opticalsurface>
# </solids>
#%%
# <structure>
#   &cam_volume;
#   <volume name="lar_volume">
#     <materialref ref="G4_lAr"/>
#     <solidref ref="lar_solid"/>
#check the sides in which the cameras are on (now x is the long side)
#this should be ok!
#       <physvol name="CAM_NW_X0_X0" copynumber="0">
#         <volumeref ref="cam_volume"/>
#         <position name="pos" x="0" y="0" z="distance/2 + assembly_thick / 2"/>
#         <rotation name="Rotation_0" unit="deg" x="0" y="180" z="0"/>
#       </physvol>

#       <physvol name="CAM_NE_X0_Y0" copynumber="1">
#         <volumeref ref="cam_volume"/>
#         <position name="pos" x="0" y="0" z="-distance/2 - assembly_thick / 2"/>
#       </physvol>
      
#     <auxiliary auxtype="Fiducial" auxvalue=""/>
#   </volume>


#    <volume name="air_layer_volume">
#      <materialref ref="G4_AIR"/>
#      <solidref ref="air_layer"/>
#        <physvol name="lar_physical">
#          <volumeref ref="lar_volume"/>
#        </physvol>
#    </volume>

#   <volume name="world_small_volume">
#     <materialref ref="G4_AIR"/>
#     <solidref ref="world_small"/>
#       <physvol name="air_layer_physical">
#         <volumeref ref="air_layer_volume"/>
#       </physvol>
#   </volume>
#aggiungo un volume d'aria per definire la superficie di assorbimento
#   <bordersurface name="AirBoxSurface" surfaceproperty="opSurfdewar_solid">
#     <physvolref ref="lar_physical"/>
#     <physvolref ref="air_layer_physical"/>
#   </bordersurface>
# </structure>

# <setup name="Default" version="1.0">
#   <world ref="world_small_volume"/>
# </setup>

# </gdml>
#%% The code template
temp='''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE gdml [
<!ENTITY properties SYSTEM "./properties.xml">
<!ENTITY cam_volume SYSTEM "./cam_volume.xml">
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

  <material name="G4_lAr" state="liquid">
    <property name="RINDEX" ref="LAr_RINDEX"/>
    <property name="GROUPVEL" ref="LAr_GROUPVEL"/>
    <property name="RAYLEIGH" ref="LAr_RAYLEIGH"/>
    <property name="ABSLENGTH" ref="LAr_ABSLENGTH"/>
    <property name="FASTCOMPONENT" ref="LAr_FASTCOMPONENT"/>
    <property name="SLOWCOMPONENT" ref="LAr_SLOWCOMPONENT"/>
    <property name="SCINTILLATIONYIELD" ref="LAr_SCINTILLATIONYIELD"/>
    <property name="RESOLUTIONSCALE" ref="LAr_RESOLUTIONSCALE"/>
    <property name="FASTTIMECONSTANT" ref="LAr_FASTTIMECONSTANT"/>
    <property name="SLOWTIMECONSTANT" ref="LAr_SLOWTIMECONSTANT"/>
    <property name="YIELDRATIO" ref="LAr_YIELDRATIO"/>
    <T unit="K" value="293.15"/>
    <MEE unit="eV" value="188"/>
    <D unit="g/cm3" value="1.396"/>
    <fraction n="1" ref="Ar_element"/>
  </material>
</materials>
<solids>
  <box lunit="mm" name="cam_solid" x="{0:.03f}+10" y="{0:.03f}+10" z="{1:.03f}+10"/>
  <box lunit="mm" name="lar_solid" x="{2:.03f}" y="{2:.03f}" z="{3:.03f}+2*{1:.03f}+10"/>
  <box lunit="mm" name="air_layer" x="{2:.03f}+0.1" y="{2:.03f}+0.1" z="{3:.03f}+2*{1:.03f}+0.1+10"/>
  <box lunit="mm" name="world_small" x="{2:.03f}+0.2" y="{2:.03f}+0.1" z="{3:.03f}+2*{1:.03f}+0.2+10"/>
  <opticalsurface finish="0" model="0" name="opSurfdewar_solid" type="1" value="1">
    <property name="REFLECTIVITY" ref="meniscus_surface_REFLECTIVITY"/>
    <property name="EFFICIENCY" ref="meniscus_surface_EFFICIENCY"/>
  </opticalsurface>
</solids>
<structure>
  &cam_volume;
  <volume name="lar_volume">
    <materialref ref="G4_lAr"/>
    <solidref ref="lar_solid"/>
    
      <physvol name="CAM_NW_X0_X0" copynumber="0">
        <volumeref ref="cam_volume"/>
        <position name="pos" x="0" y="0" z="{3:.03f}/2 + {1:.03f} / 2"/>
        <rotation name="Rotation_0" unit="deg" x="0" y="180" z="0"/>
      </physvol>

      <physvol name="CAM_NE_X0_Y0" copynumber="1">
        <volumeref ref="cam_volume"/>
        <position name="pos" x="0" y="0" z="-{3:.03f}/2 - {1:.03f} / 2"/>
      </physvol>
      
    <auxiliary auxtype="Fiducial" auxvalue=""/>
  </volume>
  
    <volume name="air_layer_volume">
      <materialref ref="G4_AIR"/>
      <solidref ref="air_layer"/>
        <physvol name="lar_physical">
          <volumeref ref="lar_volume"/>
        </physvol>
    </volume>

  <volume name="world_small_volume">
    <materialref ref="G4_AIR"/>
    <solidref ref="world_small"/>
      <physvol name="air_layer_physical">
        <volumeref ref="air_layer_volume"/>
      </physvol>
  </volume>
  
  <bordersurface name="AirBoxSurface" surfaceproperty="opSurfdewar_solid">
    <physvolref ref="lar_physical"/>
    <physvolref ref="air_layer_physical"/>
  </bordersurface>
</structure>

<setup name="Default" version="1.0">
  <world ref="world_small_volume"/>
</setup>

</gdml>
'''

#%% Making of the cold_demonstrator.gdml file
def make_cold_dem(codedApertureMask_side,assembly_thick,side_width,distance,path):
    with open('{0}/cold_demonstrator.gdml'.format(path), 'w') as f:
        print(temp.format(codedApertureMask_side,assembly_thick,side_width,distance),file=f)





