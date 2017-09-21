"""Functions to load map."""
import numpy as np
from os.path import abspath, dirname, exists, join
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

    for v in paths.values():
        assert(exists(v))

    return paths


def load_map(imname):
    """Load in a certain obstacle map as 0s and 1s.

    Everywhere the map is passable should be a 1, everywhere
    it is impassable should be a zero.
    """
    p = get_paths()
    impath = join(p['data'], imname)
    im = imread(impath, flatten=True, mode='L')
    if imname in ['maze.jpg']:
        im[im <= 150] = 0  # passable
        im[im > 150] = 1  # impassable
    return im


def get_track(start_position, bearing, jump):
    """Get pixel ids corresponding to a track."""
    dr = np.arange(0, jump, 0.1)  # increment tracks by 0.1 pixel to be exhaustive
    track = np.array([start_position + r * np.array([np.cos(bearing), np.sin(bearing)])
                      for r in dr])
    track = track.astype('int')
    return track


def is_obstructed(state: np.ndarray, jump: float, map: np.ndarray) -> bool:
    """Evaluate whether the an object has attempted to go through a wall.

    Parameters
    ----------
    state: np.ndarray
        numpy array containing [bearing, x position, y position], representing
        the initial state of the object (bearing is in radians)
    jump: float
        distance to jump (in pixels)
    map: np.ndarray
        map over which particle moves, of shape (m, n), containing 0s for passable
        elements and 1 for impassable elements (walls, obstacles, etc.)

    Returns
    -------
    blocked: bool
        whether or not the input particle would be obstructed by impassable elements
        in the map in attempting to traverse from its starting position along its bearing
        by specified jump distance
    """
    # Get list of all the pixels the particle would have to travel through
    # Account for the possibility of hitting the corner of a pixel by perturbing
    # two starting points a small amount perpendicular to its bearing and creating
    # two parallel pixel tracks from these starting points
    eps = 0.01
    bearing = state[0]
    xy_start_upper = state[1:] + eps * np.array([-np.sin(bearing), np.cos(bearing)])
    xy_start_lower = state[1:] - eps * np.array([-np.sin(bearing), np.cos(bearing)])

    track_upper = np.array([xy_start_upper + r * np.array([np.cos(bearing), np.sin(bearing)]) for r in dr])
    track_lower = np.array([xy_start_lower + r * np.array([np.cos(bearing), np.sin(bearing)]) for r in dr])

    # convert all entries to integers
    track_upper = track_upper.astype('int')
    track_lower = track_lower.astype('int')


    return track_upper, track_lower