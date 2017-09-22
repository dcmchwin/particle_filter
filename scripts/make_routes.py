import numpy as np
import os.path as op
import particle.map as mp
import particle.filter as fl

# load map and generate route
le_map = mp.load_map('maze01.jpg')
route = fl.generate_route(le_map)

# save route
routename = 'maze01_route.npy'
routepath = op.join(mp.get_paths()['routes'], routename)
np.save(routepath, route)
