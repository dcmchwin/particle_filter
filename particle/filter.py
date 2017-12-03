"""Functions for particle filter of simple 2D robot."""
import logging
import numpy as np
from typing import List

logger = logging.Logger(__file__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def predict(state_in, u, Q, le_map, dt=1.0):
    """Predict next stage of robot movement (turn, then jump).

    Parameters
    ----------
    state_in: np.ndarray
        input state, [[bearing in radians, x pos, y pos, is blocked]]
    u: np.ndarray
        process model for movement [angular velocity, linear speed]
    Q: np.ndarray
        uncertainty on process model
        [angular velocity uncertainty, linear speed uncertainty]
    le_map: np.ndarray
        array of 'impassability' booleans describing the zone to traverse,
        showing where walls and other obstacles are
    dt: float, optional
        time step

    Returns
    -------
    state_out: np.ndarray
        output state

    """
    # account for input dimensionality
    if np.ndim(state_in) == 1:
        state_in = state_in[np.newaxis, :]

    x = state_in.copy()

    # Generate random elements of update in bearing change and speed
    q0 = Q[0] * np.random.rand(state_in.shape[0])
    q1 = Q[1] * np.random.randn(state_in.shape[0])

    # Get jumps
    dphi = dt * (u[0] + q0)
    dr = dt * (u[1] + q1)

    # Use jumps to update state
    # Turn
    x[:, 0] = state_in[:, 0] + dphi
    x[:, 0] %= 2 * np.pi

    # Evaluate whether or not any particles have performed an impossible journey
    # (i.e. been blocked by a wall)
    for i in range(np.shape(x)[0]):
        x[i, 3] = is_obstructed(x[i, :3], dr[i], le_map)

    # Jump along new bearing
    x[:, 1] = x[:, 1] + dr * np.cos(x[:, 0])
    x[:, 2] = x[:, 2] + dr * np.sin(x[:, 0])
    x[:, 1:3] = x[:, 1:3].round().astype('int')

    state_out = x

    return state_out


def get_track(start_position, bearing, jump):
    """Get pixel ids corresponding to a track."""
    dr = min(0.1, 0.5 * jump)
    steps = np.arange(0, jump + dr, dr)  # increment tracks by 0.1 pixel to be exhaustive
    track = np.array([start_position + r * np.array([np.cos(bearing), np.sin(bearing)])
                      for r in steps])
    track = track.round().astype('int')
    track = np.unique(track, axis=0)

    return track


def is_obstructed(state: np.ndarray, jump: float, le_map: np.ndarray) -> bool:
    """Evaluate whether the an object has attempted to go through a wall.

    Parameters
    ----------
    state: np.ndarray
        numpy array containing [[bearing, x position, y position]], representing
        the initial state of the object (bearing is in radians)
    jump: float
        distance to jump (in pixels)
    le_map: np.ndarray
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
    eps = 0.05
    bearing = state[0]
    xy_start_upper = state[1:] + eps * np.hstack([-np.sin(bearing), np.cos(bearing)])
    xy_start_lower = state[1:] - eps * np.hstack([-np.sin(bearing), np.cos(bearing)])

    track_upper = get_track(xy_start_upper, bearing, jump)
    track_lower = get_track(xy_start_lower, bearing, jump)

    # combine the parallel tracks
    track = np.unique(np.vstack((track_upper, track_lower)), axis=0)

    # if track goes out of bounds, then it must be blocked
    m, n = np.shape(le_map)
    if (track < 0).any() or (track[:, 0] >= m).any() or (track[:, 1] >= n).any():
        blocked = True
    else:
        # get the passability values from the map
        track_blocked = le_map[track[:, 0], track[:, 1]]
        if track_blocked.any():
            blocked = True
        else:
            blocked = False

    return blocked


def generate_route(le_map, n_step=1000, v=None, dt=1):
    """Generate a route for a robot through a given map.

    Parameters
    ----------
    le_map: np.ndarray
    n_step: int
    v: float
    dt: float

    Returns
    -------
    state_history:
    process_model:
    Q:
    dt:

    """
    height, width = np.shape(le_map)  # extent of map

    # default speed is 50th of the map size
    if v is None:
        v = max(height / 20, width/20, 5)

    # define process noise
    Q = np.array([0.08 * np.pi, 0.1 * v])

    # initialise empty motion observations array
    process_model = np.empty([n_step - 1, 2])

    # init random starting position, ensuring we don't start inside a wall
    start_passable = False
    while not start_passable:
        x0 = ([height, width] * np.random.rand(2)).astype('int')
        start_passable = not le_map[x0[0], x0[1]]

    # get random starting bearing
    bearing = np.pi * np.random.rand()
    gone_through_wall = False
    state = np.array([[bearing, x0[0], x0[1], gone_through_wall]])

    # init state history with first state
    state_history = np.empty((n_step, np.shape(state)[1]))
    state_history[0, :] = state

    # move the particle randomly about the space
    i_step = 0
    while i_step < n_step - 1:
        u = np.array([0.4 * np.pi * np.random.randn(), v])
        candidate_state = predict(state, u, Q, le_map, dt)
        if not candidate_state[:, 3].any():
            state = candidate_state
            state_history[i_step + 1, :] = candidate_state[0, :]
            process_model[i_step, :] = u
            i_step = i_step + 1

    return state_history, process_model, Q, dt


def update(pop_in):
    """Update the particle states across the whole population.

    Assumes that the 'predict' state has just been wrong. Process is to:
    1. Remove all particles that have gone somewhere impassable
    2. Replace them with copies of a random draw of the remaining particles

    Parameters
    ----------
    pop_in: np.ndarray
        n x 4 array of particles

    Returns
    -------
    pop_out: np.ndarray
        n x 4 array of particles
    """
    pop_out = pop_in.copy()

    # get indices of particles to remove, and which ones to use as replacement
    is_blocked = pop_in[:, 3]
    idx_to_remove = np.nonzero(is_blocked)[0]
    idx_remaining = np.nonzero(-is_blocked + 1)[0]

    if len(idx_to_remove) == 0:
        pass
    elif len(idx_remaining) == 0:
        logger.error("All particles obstructed")
        raise ValueError("All particles obstructed")
    else:
        idx_replace = np.random.choice(idx_remaining, idx_to_remove.shape)

        # do resampling: replace all excludable particles with a random
        # choice of remaining allowed particles
        pop_out[idx_to_remove, :] = pop_in[idx_replace, :]

    return pop_out


if __name__ == '__main__':
    state = np.array([0.0 * np.pi, 0, 0])
    msz = 100
    le_map = np.random.rand(msz, msz) < 0.02
    route = generate_route(le_map)
