from gtsam import Point3, Pose3
import plotly.express as px
import numpy as np
import gtsam
import math

import matplotlib.pyplot as plt
from gtsam.utils import plot

NM = gtsam.noiseModel

# load the odometry
# DR: Odometry Input (delta distance traveled and delta heading change)
#    Time (sec)  Delta Distance Traveled (m) Delta Heading (rad)
odometry = {}
path_to_odo = '/'
data_file = 
for row in np.loadtxt(data_file):
    t, distance_traveled, delta_heading = row
    odometry[t] = Pose3(distance_traveled, 0, delta_heading)
M = len(odometry)
print(f"Read {M} odometry entries.")