import numpy as np
from skimage.io import imread, imsave
from os import walk
from os.path import join
import sys

if (len(sys.argv) < 2):
    raise RuntimeError("Not enough argument")

for r, _, files in walk(sys.argv[1]):
    for f in files:
        if f.endswith('.jpg'):
            try:
                file_name = join(r, f)
                m = imread(file_name)
                if m.shape[0] > m.shape[1]:
                    m = np.rot90(m)
                    imsave(file_name, m)
            except Exception:
                print('File {} got problem'.format(join(r, f)))
