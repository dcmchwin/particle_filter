"""Functions to load map."""
import numpy as np
from os.path import abspath, dirname, exists, join, splitext
from scipy.misc import imread
from typing import Dict, List


def get_paths():
    """Get useful paths for this project."""
    tf = abspath(__file__)
    paths = dict()
    paths['package'] = dirname(tf)
    paths['project'] = dirname(paths['package'])
    paths['notebooks'] = join(paths['project'], 'notebooks')
    paths['data'] = join(paths['project'], 'data')
    paths['images'] = join(paths['data'], 'images')
    paths['routes'] = join(paths['data'], 'routes')

    for v in paths.values():
        assert(exists(v))

    return paths


def load_map(imname):
    """Load in a certain obstacle map as 0s and 1s.

    Everywhere the map is passable should be a 1, everywhere
    it is impassable should be a zero.
    """
    p = get_paths()
    impath = join(p['images'], imname)
    im = imread(impath, flatten=True, mode='L')
    if splitext(imname)[1] in ['.jpg']:
        im = im < 150
    return im
