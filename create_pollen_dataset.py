import pandas as pd
from slide_generator import ObjectSlidePlanner, ObjectSlideGenerator, tighten_bounding_boxes
import os

pollen_images = pd.read_csv('pollen_images.csv')
artefact_images = pd.read_csv('artefact_sizes.csv')
pollen_sizes = pd.read_csv('pollen_sizes.csv')

object_name = 'test'
slide_dataframe_name = f'{object_name}.csv'
fname = f'{object_name}/{object_name}'
mask_fname = f'{object_name}_masks/{object_name}_mask'
os.mkdir(object_name)
os.mkdir(f'{object_name}_masks')

slide_dims = (1280, 1280, 3)

n_slides = 10 #10000
pollen_per_slides = (1, 10)
artefacts_per_slide = (1, 10)

#class_split = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0) # Corylus
#class_split = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0) # Secale
#class_split = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0) # Urtica
#class_split = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0) # Chenopodium

class_split = (.25, 0, .25, .25, 0, 0, .25, 0, 0, 0) # equal split
#class_split = (.1, 0, .7, .1, 0, 0, .1, 0, 0, 0) # bigger pollen split
#class_split = (.1, 0, .1, .7, 0, 0, .1, 0, 0, 0) # small pollen split
#class_split = (.4, 0, .25, .25, 0, 0, .1, 0, 0, 0) # middle pollen split

gen = ObjectSlidePlanner(pollen_images, pollen_sizes, artefact_images, slide_dims)
gen.create_slide_structure(n_slides, pollen_per_slides, artefacts_per_slide, class_split, slide_dataframe_name)

slide_df = pd.read_csv(slide_dataframe_name)

gen = ObjectSlideGenerator(pollen_images, artefact_images, slide_dims, slide_df, fname, mask_fname)
gen.generate()
gen.generate_masks()
tighten_bounding_boxes([object_name])