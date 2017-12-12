"""Functions for particle filter of simple 2D robot."""
import logging
import numpy as np
from typing import List

logger = logging.Logger(__file__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
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
    jump = dt * (u[1] + q1)

    # Use jumps to update state
    # Turn
    x[:, 0] = state_in[:, 0] + dphi
    x[:, 0] %= 2 * np.pi

    # Evaluate whether or not any particles have performed an impossible journey
    # (i.e. been blocked by a wall)
    x[:, 3] = is_obstructed(x[:, :3], jump, le_map)

    # Jump along new bearing
    x[:, 1] = x[:, 1] + jump * np.cos(x[:, 0])
    x[:, 2] = x[:, 2] + jump * np.sin(x[:, 0])
    x[:, 1:3] = x[:, 1:3].round().astype('int')

    state_out = x

    return state_out


def get_single_track(start_position, bearing, jump):
    """Get pixel ids corresponding to a track.

    Parameters
    ----------
    start_position: np.ndarray
        [n_particles, n_dimensions] array of track start positions
    bearing: np.ndarray
        [n_particles] array of track bearings for each particle
    jump: np.ndarray
        [n_particles] array of track lengths for each particle

    Returns
    -------
    track: np.ndarray
        [n_particles, n_dimensions, n_pixels] array of track start positions
        as integers
    """
    # ensure that maximum step size is 0.3, or half the total jump distance
    min_increment_scalar = 0.3
    dr = np.where(jump > 2 * min_increment_scalar, min_increment_scalar, 0.5 * jump)
    n_steps = np.ceil(max(jump / dr)).astype('int')

    # Get 2d array of radial distances away from start position for each particle
    # This array is of shape (nParticles, nSteps)
    r = jump[:, np.newaxis] * np.linspace(0, 1, n_steps)

    # Get direction vectors to move along for each particle
    # This array is of shape (nParticles, nDims), nDims = 2 for x and y
    direction_vectors = np.hstack([np.cos(bearing[:, np.newaxis]),
                                   np.sin(bearing[:, np.newaxis])])

    # Get the track
    # The shape of this array is (nParticles, nDims, nSteps)
    relative_track = r[:, np.newaxis, :] * direction_vectors[:, :, np.newaxis]
    track = start_position[:, :, np.newaxis] + relative_track

    # In getting unique elements, the number of steps will reduce
    track = np.unique(track.round().astype('int'), axis=2)

    return track


def get_parallel_tracks(state: np.ndarray, jump: np.ndarray):
    """Get two sets of parallel particle tracks.

    Parameters
    ----------
    state: np.ndarray
        numpy array containing [[bearing, x position, y position]], representing
        the initial state of the object (bearing is in radians). Shape of this
        array is (nParticles, 3)
    jump: np.ndarray
        distance to jump (in pixels) for each particle. Shape of this array
        is (nParticles,)

    Returns
    -------
    track_upper: np.ndarray
        upper tracks for each particle in state. Shape of this array is
        (nParticles, nDims, nSteps_upper)
    track_lower: np.ndarray
        lower tracks for each particle in state. Shape of this array is
        (nParticles, nDims, nSteps_lower)
    """
    bearing = state[:, 0]

    direction_vectors = np.hstack([-np.sin(bearing[:, np.newaxis]),
                                   np.cos(bearing[:, np.newaxis])])

    eps = 0.05
    xy_start_upper = state[:, 1:] + eps * direction_vectors
    xy_start_lower = state[:, 1:] - eps * direction_vectors

    track_upper = get_single_track(xy_start_upper, bearing, jump)
    track_lower = get_single_track(xy_start_lower, bearing, jump)

    return track_upper, track_lower


def combine_tracks(track_upper: np.ndarray, track_lower: np.ndarray):
    """Combine two parallel tracks.

    Parameters
    ----------
    track_upper: np.ndarray
        upper tracks for each particle in state. Shape of this array is
        (nParticles, nDims, nSteps_upper)
    track_lower: np.ndarray
        lower tracks for each particle in state. Shape of this array is
        (nParticles, nDims, nSteps_lower)

    Returns
    -------
    track: np.ndarray
        tracks for each particle in state. Shape of this array is
        (nParticles, nDims, nSteps_lower + nSteps_upper)
    """
    # combine the parallel tracks
    track = np.concatenate((track_upper, track_lower), axis=2)
    return track


def is_obstructed(state: np.ndarray, jump: np.ndarray, le_map: np.ndarray) -> bool:
    """Evaluate whether the an object has attempted to go through a wall.

    Parameters
    ----------
    state: np.ndarray
        numpy array containing [[bearing, x position, y position]], representing
        the initial state of the object (bearing is in radians). Shape of this array
        is (nParticles, 3)
    jump: np.ndarray
        distance to jump (in pixels) for each particle. Shape of this array is (nParticles,)
    le_map: np.ndarray
        map over which particle moves, of shape (m, n), containing 0s for passable
        elements and 1 for impassable elements (walls, obstacles, etc.)

    Returns
    -------
    blocked: np.ndarray[bool]
        whether or not the input particles would be obstructed by impassable elements
        in the map in attempting to traverse from their starting positions along their
        respective bearings by specified the jump distance. Shape of this array is (nParticles,)
    """
    # Get list of all the pixels the particle would have to travel through
    # Account for the possibility of hitting the corner of a pixel by perturbing
    # two starting points a small amount perpendicular to its bearing and creating
    # two parallel pixel tracks from these starting points
    track_upper, track_lower = get_parallel_tracks(state, jump)

    # combine the parallel tracks
    track = combine_tracks(track_upper, track_lower)

    ## Now use tracks to calculate 'blockedness'

    # Make a duplicate map that is padded with 'impassable values' so that we
    # can do an all-in-one 'gone off map or gone through wall' calculation
    pad_width = int(max(jump)) + 1
    le_map_padded = np.pad(le_map, pad_width,
                           'constant', constant_values=True)

    # We have to update the indices of the tracks to account for the pad width
    track_padded = track + pad_width

    blocked = le_map_padded[track_padded[:, 0, :],
                            track_padded[:, 1, :]].any(axis=1)

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


def generate_particles(le_map: np.ndarray, n_particles: int) -> np.ndarray:
    """Generate starting positions of particle population.

    Parameters
    ----------
    le_map: np.ndarray
        map over which particle moves, of shape (m, n), containing 0s for passable
        elements and 1 for impassable elements (walls, obstacles, etc.)
    n_particles: int
        number of particles to generate

    Returns
    -------
    pop: np.ndarray
        population of particle starting states, of shape (n_particles, 4)

    """
    (height, width) = le_map.shape

    # particle states have 4 entries: [bearing in radians, x pos, y pos, is blocked]
    pop = np.empty((n_particles, 4))

    pop[:, 0] = 2 * np.pi * np.random.rand(n_particles)
    pop[:, 1] = np.random.randint(0, height, size=n_particles)
    pop[:, 2] = np.random.randint(0, width, size=n_particles)
    pop[:, 3] = np.zeros(n_particles)

    # Reset starting positions that land inside walls
    def is_blocked(): return le_map[pop[:, 1].astype('int'), pop[:, 2].astype('int')]

    while is_blocked().any():
        blocked = is_blocked()
        idx_blocked = blocked.nonzero()
        pop[idx_blocked, 1] = np.random.randint(0, height, blocked.sum())
        pop[idx_blocked, 2] = np.random.randint(0, width, blocked.sum())

    return pop


def compute_state_estimate(population: np.ndarray) -> np.ndarray:
    """Get the mean and standard deviation of particle positions

    Parameters
    ----------
    population: np.ndarray
        Population of particles, with predicted positions

    Returns
    -------
    state_estimate: np.ndarray
        Mean of particle states
    state_deviation: np.ndarray
        Standard deviation of particle states

    """
    state_estimate = np.median(population, axis=0)
    state_deviation = np.std(population, axis=0)
    return state_estimate, state_deviation


if __name__ == '__main__':
    state = np.array([0.0 * np.pi, 0, 0])
    msz = 100
    le_map = np.random.rand(msz, msz) < 0.02
    route = generate_route(le_map)
