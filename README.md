# Slide generator

Simple Python class for generating images for segmentation/box regression and classification from images and monochrome masks.

Csv structure used in this example simulating pollen microscope slides:

### pollen_images.csv
path,mask_paths,height,width,class_id

### pollen_sizes.csv
min_factor,max_factor
- in order of class id
- min and max for RNG used to randomly resize image

### artefact_sizes.csv
path,mask_path,height,width,min_factor,max_factor
- since there are not too many artefacts, there is no seperate file for the size factors

Example dataset that was generated using this script can be viewed at .

Sebastian Seurig (sebastian@seurig.com)
https://www.researchgate.net/profile/Sebastian_Seurig
ORCID 0000-0001-6511-0102