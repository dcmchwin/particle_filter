"""Unit tests on filter."""

import logging
import numpy as np
import os.path as op
import particle.filter as fl
import particle.map as mp
import pytest


logger = logging.Logger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

@pytest.fixture(scope='module')
def le_map():
    le_map = mp.load_map('maze01.jpg')
    return le_map

@pytest.fixture(scope='module')
def route():
    routename = 'maze01_route02.npz'
    routepath = op.join(mp.get_paths()['routes'], routename)
    route_items = np.load(routepath)
    route = route_items['state_history']
    return route

# def test_is_obstructed_scalar(le_map, route):
#     jump = 3.0
#     blocked = fl.is_obstructed(route[0, :3], jump, le_map)
#     logging.info(blocked)
#     # assert False

def test_is_obstructed(le_map, route):
    nParticles = 15
    jump = np.array([3.0] * nParticles)
    state = route[:nParticles, :3]

    blocked = fl.is_obstructed(state, jump, le_map)
    logging.debug(blocked)
    assert False


# def test_get_track(route):
#     nParticles = 10
#     jump = np.array([2.0] * nParticles)
#     eps = 0.05
#     bearing = route[:nParticles, 0]
#
#     direction_vectors = np.hstack([-np.sin(bearing[:, np.newaxis]),
#                                    np.cos(bearing[:, np.newaxis])])
#
#     a = route[:nParticles, 1:3]
#     b = eps * direction_vectors
#
#     xy_start_upper = a + b
#
#     # logging.debug('bearing: {}'.format(bearing))
#     track = fl.get_track(xy_start_upper, bearing, jump)
#
#     logger.debug('Tracks: {}'.format(track))
