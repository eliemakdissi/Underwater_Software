import threading
from gtsam import Point3, Pose3
import plotly.express as px
import numpy as np
import gtsam
import math

import matplotlib.pyplot as plt
from gtsam.utils import plot
from numpy.random import default_rng
from gtsam import symbol_shorthand
rng = default_rng()

NM = gtsam.noiseModel


class Backend():

    def __init__(self, params_stereo_):
        # parameters
        self.minPose = 1  # minimum number of keyframes to process initially
        self.incK = 1  # minimum number of keyframes to process after
        self.K = 0
        robust = True
        # Set Noise parameters
        priorSigmas = gtsam.Point3(1, 1, math.pi)
        odoSigmas = gtsam.Point3(0.05, 0.01, 0.1)
        sigmaR = 100        # range standard deviation

        self.priorNoise = NM.Diagonal.Sigmas(priorSigmas)  # prior
        self.looseNoise = NM.Isotropic.Sigma(2, 1000)     # loose LM prior
        self.odoNoise = NM.Diagonal.Sigmas(odoSigmas)     # odometry
        self.gaussian = NM.Isotropic.Sigma(1, sigmaR)     # non-robust
        self.stereo_noise = NM.Isotropic.Sigma(3, 1.0) # 1 pixel std dev (ul, ur, v)


        # Initialize iSAM
        self.isam = gtsam.ISAM2()

        self.odometry_to_add = [] #j, i, StereoPoint2
        self.landmarks_to_add = {}
        self.initializedLandmarks = set()

        self.last_pose = Pose3(0, 0, 0, 0, 0, 0)
        self.initialized = False
        self.pose_counter = 0
        self.newFactors = gtsam.NonlinearFactorGraph()
        self.newFactors.addPriorPose3(0, self.lastpose, self.priorNoise)
        self.initial = gtsam.Values()
        self.initial.insert(0, self.last_pose)
        print(self.newFactors, self.initial)

        print(self.isam)
    def insert_keyframe(self,frame):
        pass

    def optimization_step(self):
        # Loop over odometry
        for t, relative_pose in self.odometry_to_add.items():
            # add odometry factor
            self.newFactors.add(gtsam.BetweenFactorPose3(self.pose_counter - 1, self.pose_counter, relative_pose,
                                                    self.odoNoise))

            # predict pose and add as initial estimate
            predictedPose = self.lastPose.compose(relative_pose)
            self.lastPose = predictedPose
            self.initial.insert(self.pose_counter, predictedPose)
            self.K +=1

            # Check if there are landmarks to be added
        for landmark in self.landmarks_to_add:
            if landmark[1] == self.pose_counter:
                j,_,stereo_point = landmark
                landmark_key = gtsam.symbol('L', j)
                ###
                self.newFactors.add(gtsam.GenericStereoFactor3D(stereo_point,self.stereo_noise,
                    self.pose_counter, landmark_key, self.calibration))
                if landmark_key not in self.initializedLandmarks:
                    p = rng.normal(loc=0, scale=100, size=(3,)) #à modifier avec les guess du reste
                    self.initial.insert(landmark_key, p)
                    print(f"Adding landmark L{j}")
                    self.initializedLandmarks.add(landmark_key)
                    # We also add a very loose prior on the landmark in case there is only
                    # one sighting, which cannot fully determine the landmark.
                    self.newFactors.add(gtsam.PriorFactorPoint3(
                        landmark_key, Point3(0,0, 0), self.looseNoise))
                
        # Check whether we've seen enough keyframes to update the graph
        if (self.pose_counter > self.minPose) and (self.K > self.incK):
            if not initialized:  # Do a full optimize for first minK ranges
                print(f"Initializing at pose {self.pose_counter}")
                batchOptimizer = gtsam.LevenbergMarquardtOptimizer(
                    self.newFactors, self.initial)
                initial = batchOptimizer.optimize()
                initialized = True
            self.isam.update(self.newFactors, self.initial)
            result = self.isam.calculateEstimate()
            self.lastPose = result.atPose3(self.pose_counter)
            self.newFactors = gtsam.NonlinearFactorGraph()
            self.initial = gtsam.Values()
            self.K = 0
            self.odometry_to_add.pop(t)

        self.Result = self.isam.calculateEstimate()

    def optimization_loop(self):
        pass

    def plot(self):
        # plot positions
        poses = gtsam.utilities.allPose2s(self.Result)
        landmarks = gtsam.utilities.extractPoint2(self.Result)
        positions = np.array([poses.atPose2(key).translation()
                            for key in poses.keys()])
        print(positions.shape)
        fig = px.scatter(x=positions[:,0],y=positions[:,1])
        fig.add_scatter(x=landmarks[:,0], y=landmarks[:,1], mode="markers", showlegend= False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.show()
