# Geometry
This repository contains all the geometry used with the [Optical simulation](https://baltig.infn.it/dune/sand-optical/opticalmeniscus). All the geometries are stored in different branches, one per geometry.    
Here, all the details on how to write a functional geometry are given. These are all the specific names, patterns, volume and so on needed to run a simulation. No info on the gdml/xml languages are given. To learn more details on gdml/xml and how to write a geometry for geant4, look [here](https://gdml.web.cern.ch/GDML/doc/GDMLmanual.pdf).

## Geometry structure
The geometry is written in a gdml file(s). While the structure of the file is not mandatory, all the geometries here are written following the same idea: a nested structure of files, one per piece, with some values and names hardcoded. These must be set correctly and a description of each one is given later in this readme.
 
The details of each file are as follow:
* `properties.xml` - this file includes all the physical properties of the materials, such as light yield, refraction index and so on. It doesn't includes the chemical properties, such as composition, isotopes and so on. The file is included in very file of the repository and should contains the info for all the materials.

* `materials.xml` - this file includes all the definition of all the materials: composition, isotopes, density and so on. It doesn't includes the physical properties, such as light yield and refraction index. The file is included in every file of the repository and should contains the info for all the materials.

* `main.gdml` - the starting file, the one given as input to the simulation. It contains a single volume, the World volume, with a single daugther physical volume inside it defined in the "file" field. The name of this file can change depending on the geometry but will always be exactly one.

* `meniscus_curved.gdml` - this is the file used inside the World volume. Here different volume are defined, the hierarchy is written with the inner volume first, and outer volumes following it. Here the volumes:
    * `lar_volume`, the first and most important volume of this file. This is the argon volume where all the cameras are placed. Its `material should always be Argon` and its `name must always be lar_volume`. This volume should always have this auxilary value defined:
        ```xml
        <auxiliary auxtype="Fiducial" auxvalue=""/>
        ```
        Inside of it, multiple physVol are defined, one per camera. The geometry of these cameras is defined in the `cam_volume.xml` file. The name of each camera must be `unique` and must contains `"CAM_"` (without quotation marks) in the name.
    * `vessel_int_volume`, the inner vessel of the cryostat. It follow the lar_volume in the hyerarchy. This should include the lar_volume as a physical volume but nothing else is mandatory.

    * `air_volume`, the volume between the vessels. It follow the vessel_int_volume in the hyerarchy. This should include the vessel_int_volume as a physical volume but nothing else is mandatory.

    * `vessel_ext_volume`, the external vessel of the cryostat. It follow the air_volume in the hyerarchy. This should include the air_volume as a physical volume but nothing else is mandatory.

* `cam_volume.xml` - this is the file used inside each "CAM_" physical volume of the lar_volume. Here, the volume of a single camera is defined. This volume has three pysical volumes:
    * `cameraAssembly_mask` is the mask volume. Its volume is defined in the codedApertureMask.gdml file.
    * `cameraAssembly_photoDetector` is the sensor volume. Its volume is defined in the photoDetector.gdml file.
    * `cameraAssembly_body` is the body volume (the mechanical support). Its volume is defined in the cameraBody.gdml file.

* `codedApertureMask.gdml` - this is the file where the volume of the coded mask is defined. The solid mask is obtained with a sequence of subtractions in the structure section. This is not doable by hand and should be generated with some software (like [this](https://baltig.infn.it/dune/sand-optical/tools/-/tree/main/gdml_maker)). This file has a single volume with some mandatory auxiliary values. The values of such auxiliaries should match the one of the geometry and the names should not be changed. Here the auxiliary values:

    ```xml
    <auxiliary auxtype="Mask" auxvalue="codedApertureMask">
      <auxiliary auxtype="rank" auxvalue="17"/>
      <auxiliary auxtype="cellcount" auxvalue="33"/>
      <auxiliary auxtype="cellsize" auxunit="mm" auxvalue="3.96"/>
      <auxiliary auxtype="celledge" auxunit="mm" auxvalue="0.2"/>
    </auxiliary>
    ```
* `photoDetector.gdml` - this is the file where the volume of the sensor is defined. This file has a single volume with some mandatory auxiliary values. The values of such auxiliaries should match the one of the geometry and the names should not be changed. Here the auxiliary values:

    ```xml
    <auxiliary auxtype="Sensor" auxvalue="S14160-6050HS">
    <auxiliary auxtype="cellcount" auxvalue="32"/>
    <auxiliary auxtype="cellsize" auxunit="mm" auxvalue="3.000"/>
    <auxiliary auxtype="celledge" auxunit="mm" auxvalue="0.200"/>
    </auxiliary>
    ```
* `cameraBody.gdml` - this is the file where the volume of the body is defined. This file has a single volume and no specific constraints.

## Extra info

### Surfaces
Some files have a `bordersurface` or a `skinsurface` defined between some volumes. These are used to absorb all the optical photons that hit the corresponding volume. Here an example of both the surfaces:
```xml
<skinsurface name="cameraBody_Surface" surfaceproperty="cameraBodySurf">
  <volumeref ref="cameraBody"/>
</skinsurface>

<bordersurface name="MeniscusBoxSurface" surfaceproperty="opSurfMeniscus_solid">
  <!-- volume order matter! -->
  <physvolref ref="lar_physical"/>
  <physvolref ref="vessel_int_physical"/>
</bordersurface>
  ```

### File types
Two file type are used in these geometries: xml and gdml. The use of these files is different and they have different requirements:
* `gdml` - these files must be fully functional gdml files. All the required field must be set. These file can be included in other gdml files using the `file` field:
    ```xml
    <file name="./path/to/file.gdml"/>
    ```

* `xml` - these file are just pieces of a gdml file. The text in these file is copied in the gdml file. To do so, an include and a call must be added to the gdml:
    * `include`, this must be done at the beginning of the file, before the define section.
        ```xml
        <!DOCTYPE gdml [
        <!ENTITY variable SYSTEM "./path/to/file.xml">
        <!ENTITY otherVariable SYSTEM "./path/to/differentFile.xml">
        ]>
        ```
    * `call`, ths must be done where the code in the xml file must be copied. This is done simply by:
        ```xml
        &variable;
        &otherVariable;
        ```
        doing so, the content of `file.xml` will be copied where the `variable` call is, while the content of the `differentFile.xml` will be copied where the `otherVariable` call is.  
        materials.xml, cam_volume.xml and properties.xml use this type of include.
        These files must be considered as part of the gdml file including them, `volumes name, materials or anything else must be choerent` between the xml and gdml files (example: cam_volume.xml uses the cam_solid volume, defined in the meniscus_curved.gdml file).