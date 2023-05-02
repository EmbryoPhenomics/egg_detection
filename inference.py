import tensorflow as tf
import numpy as np
from tqdm import tqdm
from more_itertools import chunked
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import re
import vuba
import pandas as pd
import glob
import os

from models import build_model

def read_img(fn):
    img = tf.io.read_file(fn)
    img = tf.image.decode_png(img, channels=1)
    img.set_shape([None, None, 1])
    img = tf.image.resize(images=img, size=[512, 512])
    return img

def dataset(files, batch_size):
    data = tf.data.Dataset.from_tensor_slices((files))
    data = data.map(read_img, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=False)
    return data    

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Paramters ---------------
w,h = 512, 512
batch_size = 16

# Note that this script expects individual images for a given experiment in one folder
# The format for this specific script is written for my experiment structure:
# e.g. A4_A2_0.png where A4 is the genotype, A2 is the individual, and 0 is the timepoint.
# You will need to edit the string parsing below if your data follows a different structure.
source_path = '/path/to/images' # Edit to add your own path
output_path = '/path/to/output' # Edit to add path to output csv
# -------------------------

files = glob.glob(os.path.join(source_path, '*.png'))
data = dataset(files, batch_size)

model = build_model(input_shape=(w,h,1), backbone='Xception', pretrained_weights=True)

df = dict(
    filename=[], 
    genotype=[], 
    replicate=[], 
    x1=[], y1=[], x2=[], y2=[], 
    timepoint=[])

pg = tqdm(total=len(files))
for i, batch in enumerate(data):
    filenames = files[i * batch_size : (i * batch_size) + len(batch)]

    results = model.predict_on_batch(batch)

    for fn,r in zip(filenames, results):
        r = tuple(np.asarray(r*512))
        r = tuple(map(int, r)) 

        x1,y1,x2,y2 = r
        x1,y1,x2,y2 = x1-5,y1-5,x2+5,y2+5 # Expand bbox slightly
        
        # Note this is expects a certain 
        replicate = str.split(fn, '/')[-1]
        genotype = str.split(replicate, '_')[0]
        timepoint = str.split(replicate, '_')[-1]
        timepoint = int(re.sub('.png', '', timepoint)) + 1
        replicate = str.split(replicate, '_')[0:2]

        replicate = f'{replicate[0]}_{replicate[1]}'

        df['filename'].append(fn)
        df['genotype'].append(genotype)
        df['replicate'].append(replicate)
        df['x1'].append(x1)
        df['y1'].append(y1)
        df['x2'].append(x2)
        df['y2'].append(y2)
        df['timepoint'].append(timepoint)

        pg.update(1)


df = pd.DataFrame(data=df)
df.to_csv(output_path)
