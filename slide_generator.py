import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import rotate, resize
from skimage.io import imsave, imread

class ObjectSlidePlanner:
  # creates csv file for image generation

  def __init__(self, image_dataframe, object_size_dataframe, artefact_dataframe, slide_dims):
    self.object_df = image_dataframe
    self.artefact_df = artefact_dataframe
    self.object_size_dataframe = object_size_dataframe
    self.n_classes = max(self.object_df.class_id.to_list()) + 1
    self.slide_dims = slide_dims
    self.generation_hashs = []
    self.generation_timeout = 50

  def create_slide_structure(self, n_slides, object_per_slides, artefacts_per_slide, class_split, fname):
    all_objects = {'slide' : [], 'class' : [], 'x1' : [], 'y1' : [], 'x2' : [], 'y2' : [], 'height' : [], 'width' : [], 'rot' : [], 'path' : [], 'mask_path' : []}
    for i in tqdm(range(n_slides)):
      y1, x1, y2, x2, height, width, object_class, rot, path, mask_path = self.plan_slide(object_per_slides, artefacts_per_slide, class_split)
      all_objects['slide'].extend(len(object_class)*[i])
      all_objects['class'].extend(object_class)
      all_objects['x1'].extend(x1)
      all_objects['y1'].extend(y1)
      all_objects['x2'].extend(x2)
      all_objects['y2'].extend(y2)
      all_objects['height'].extend(height)
      all_objects['width'].extend(width)
      all_objects['path'].extend(path)
      all_objects['mask_path'].extend(mask_path)
      all_objects['rot'].extend(rot)
    pd.DataFrame.from_dict(all_objects).to_csv(fname, index=False)

  def slide_collision_check(self, object_single):
    px, py, width, height = object_single
    if px is None: return True
    if self.current_slide_object == []: return False

    for lx, ly, lw, lh in self.current_slide_object:
      if ((px <= lx <= px + width) or (lx <= px <= lx + width)) and ((py <= ly <= py + height) or (ly <= py <= ly + height)): return True
      if ((px <= lx + lw <= px + width) or (lx <= px + width <= lx + lw)) and ((py <= ly + lh <= py + height) or (ly <= py + height <= ly + lh)): return True
    return False

  def plan_slide(self, object_per_slides, artefacts_per_slide, class_split):
    classes, x1, y1, x2, y2, orig_width, orig_height, angles, paths, mask_paths = [], [], [], [], [], [], [], [], [], []
    
    n_object = np.random.randint(object_per_slides[0], object_per_slides[1]+1)
    n_artefacts = np.random.randint(artefacts_per_slide[0], artefacts_per_slide[1]+1)

    self.current_slide_object = []
    px, py, rot_object_width, rot_object_height = None, None, None, None

    for _ in range(n_object):
      current_hash = None
      timeout_counter = 0
      while (current_hash is None) or (current_hash in self.generation_hashs) or self.slide_collision_check([px, py, rot_object_width, rot_object_height]):
        object_class = np.random.choice(np.arange(0, self.n_classes), p=class_split)
        object = self.object_df[self.object_df.class_id==object_class].sample(n=1)
        object_path = object.path.to_list()[0]
        object_mask_path = object.mask_paths.to_list()[0]
        angle = np.random.randint(0, 360)

        natural_size = (self.object_size_dataframe.min_factor.to_list()[object_class] * min(self.slide_dims[:2]), self.object_size_dataframe.max_factor.to_list()[object_class] * min(self.slide_dims[:2]))
        size_factor = np.random.randint(natural_size[0], natural_size[1]) if natural_size[0] != natural_size[1] else natural_size[0]

        object_height = int(object.height.to_list()[0] / max(object.height.to_list()[0], object.width.to_list()[0]) * size_factor)
        object_width = int(object.width.to_list()[0] / max(object.height.to_list()[0], object.width.to_list()[0]) * size_factor)

        rot_object_height, rot_object_width, _ = rotate(np.zeros(shape=(object_height, object_width, 1)), angle, resize=True).shape

        py = np.random.randint(0, self.slide_dims[0] - rot_object_height + 1)
        px = np.random.randint(0, self.slide_dims[1] - rot_object_width + 1)
        current_hash = hash((str(object_path), int(angle), int(py), int(px)))

        timeout_counter += 1
        if timeout_counter == self.generation_timeout: break

      if timeout_counter < self.generation_timeout:
        self.current_slide_object.append([px, py, rot_object_width, rot_object_height])
        classes.append(object_class)
        angles.append(angle)
        paths.append(object_path)
        mask_paths.append(object_mask_path)
        self.generation_hashs.append(current_hash)
        x1.append(px) 
        x2.append(px + rot_object_width)
        orig_width.append(object_width)
        y1.append(py)
        orig_height.append(object_height)
        y2.append(py + rot_object_height)

    for _ in range(n_artefacts):
      current_hash = None
      while (current_hash is None) or (current_hash in self.generation_hashs) or self.slide_collision_check([px, py, rot_object_width, rot_object_height]):
        object_class = -1
        object = self.artefact_df.sample(n=1)
        object_path = object.path.to_list()[0]
        object_mask_path = object.mask_path.to_list()[0]
        min_size, max_size = object.min_factor.to_list()[0], object.max_factor.to_list()[0]
        angle = np.random.randint(0, 360)

        natural_size = (self.artefact_df.min_factor.to_list()[object_class] * min(self.slide_dims[:2]), self.artefact_df.max_factor.to_list()[object_class] * min(self.slide_dims[:2]))
        size_factor = np.random.randint(min_size*1000, max_size*1000)/1000

        object_height = int(object.height.to_list()[0] * size_factor)
        object_width = int(object.width.to_list()[0] * size_factor)

        rot_object_height, rot_object_width, _ = rotate(np.zeros(shape=(object_height, object_width, 1)), angle, resize=True).shape

        py = np.random.randint(0, self.slide_dims[0] - rot_object_height + 1)
        px = np.random.randint(0, self.slide_dims[1] - rot_object_width + 1)
        current_hash = hash((str(object_path), int(angle), int(py), int(px)))

      self.current_slide_object.append([px, py, rot_object_width, rot_object_height])
      classes.append(object_class)
      angles.append(angle)
      paths.append(object_path)
      mask_paths.append(object_mask_path)
      self.generation_hashs.append(current_hash)
      x1.append(px) 
      x2.append(px + rot_object_width)
      orig_width.append(object_width)
      y1.append(py)
      orig_height.append(object_height)
      y2.append(py + rot_object_height)

    return y1, x1, y2, x2, orig_height, orig_width, classes, angles, paths, mask_paths

class ObjectSlideGenerator:
  # created images from csv file

  def __init__(self, image_dataframe, artefact_dataframe, slide_dims, slide_df, fname, mask_fname):
    self.object_df = image_dataframe
    self.artefact_df = artefact_dataframe
    self.slide_dims = slide_dims
    self.slide_df = slide_df
    self.fname = fname
    self.mask_fname = mask_fname
    self.image_memory = {}

  def get_image(self, p, mask = False):
    if p not in self.image_memory.keys(): 
      self.image_memory[p] = imread(p)/255.
      if mask is True: 
        im = self.image_memory[p]
        im[im <= 0.5] = 0
        im[im > 0.5] = 1
        self.image_memory[p] = im
    return self.image_memory[p]
  
  def generate(self):
    for n in tqdm(range(max(self.slide_df.slide.to_list())+1)):
      slide = np.random.uniform(.8, .9, self.slide_dims)
      sub_df = self.slide_df[self.slide_df.slide==n]
      for r in range(len(sub_df)):
        obj = sub_df.iloc[r]
        height = obj.height
        width = obj.width

        im = self.get_image(obj.path)
        mask = self.get_image(obj.mask_path, mask=True)
        im = resize(im, (height, width, self.slide_dims[-1]))
        mask = resize(mask, (height, width, self.slide_dims[-1]))
        im = rotate(im, obj.rot, resize=True)
        mask = rotate(mask, obj.rot, resize=True)

        im = im * mask + np.abs(mask - 1) * np.random.uniform(.8, .9, im.shape)

        slide[obj.y1:obj.y2, obj.x1:obj.x2, :] = im                

      imsave(f'{self.fname}_{n}.jpg', np.array(slide*255, dtype=np.uint8))
  
  def generate_masks(self):
    for n in tqdm(range(max(self.slide_df.slide.to_list())+1)):
      slide = np.zeros(shape=self.slide_dims[:2])
      sub_df = self.slide_df[self.slide_df.slide==n]
      for r in range(len(sub_df)):
        obj = sub_df.iloc[r]
        if obj['class'] == -1: continue

        height = obj.height
        width = obj.width

        mask = self.get_image(obj.mask_path, mask=True)[:,:,0]
        mask = resize(mask, (height, width))
        mask = rotate(mask, obj.rot, resize=True)

        slide[obj.y1:obj.y2, obj.x1:obj.x2] = mask                

      imsave(f'{self.mask_fname}_{n}.jpg', np.array(slide*255, dtype=np.uint8))

def tighten_bounding_boxes(object_names):
  # due to rotation bounding boxes are larger than needed

    for object_name in object_names:
        df_name = f'{object_name}.csv'
        path = f'{object_name}_masks/{object_name}_mask'

        df = pd.read_csv(df_name)
        slide_n = None

        for n in tqdm(range(len(df))):
            obj = df.iloc[n]
            if slide_n != obj.slide:
                slide_n = obj.slide
                slide = imread(f'{path}_{slide_n}.jpg')/255.
            
            if obj['class'] != -1:
                y1, x1, y2, x2 = obj.y1, obj.x1, obj.y2, obj.x2
                y2 = y2 if y2 != 1280 else 1279
                x2 = x2 if x2 != 1280 else 1279

                while np.amax(slide[y1, x1:x2]) + np.amax(slide[y1:y2, x1]) + np.amax(slide[y2, x1:x2]) + np.amax(slide[y1:y2, x2]) < 4 * .8:
                    if np.amax(slide[y1, x1:x2]) < .8: y1 = y1 + 1 if y1 != 1279 else 1279
                    if np.amax(slide[y1:y2, x1]) < .8: x1 = x1 + 1 if x1 != 1279 else 1279
                    if np.amax(slide[y2, x1:x2]) < .8: y2 = y2 - 1 if y2 != 0 else 0
                    if np.amax(slide[y1:y2, x2]) < .8: x2 = x2 - 1 if x2 != 0 else 0

                df.loc[[n], ['y1']] = y1
                df.loc[[n], ['x1']] = x1
                df.loc[[n], ['y2']] = y2
                df.loc[[n], ['x2']] = x2

        df.to_csv(f'{object_name}_tight.csv', index = False)