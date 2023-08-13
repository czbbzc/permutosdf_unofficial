import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F
import torchvision

import sys
# from permuto_sdf  import PermutoSDF
import numpy as np
import time
import math
import random


#creates ray_origins and ray_dirs of sizes Nx3 given a frame
class CreateRaysModule(torch.nn.Module):
    def __init__(self, precompute_grid=False):
        super(CreateRaysModule, self).__init__()

        #precomputed things
        # self.first_time=True
        # self.width=None
        # self.height=None
        # self.probabilities=None #will get created on the first iteration and stays the same
        self.grid_dict={} #key is the nr of pixels, the val is the probabilities for choosing a pixel for a frame of that size

        self.precompute_grid=precompute_grid

    def compute_grid(self, frame):
        # print("adding CreateRaysModule grid for ", size)
        x_coord= torch.arange(frame.width).view(-1, 1, 1).repeat(1,frame.height, 1)+0.5 #width x height x 1
        y_coord= torch.arange(frame.height).view(1, -1, 1).repeat(frame.width, 1, 1)+0.5 #width x height x 1
        ones=torch.ones(frame.width, frame.height).view(frame.width, frame.height, 1)
        points_2D=torch.cat([x_coord, y_coord, ones],2).transpose(0,1).reshape(-1,3).cuda() #Nx3 we tranpose because we want x cooridnate to be inner most so that we traverse row-wise the image

        return points_2D


    def forward(self, frame, rand_indices):

        if len(self.grid_dict)>50:
            print("We have a list of grid_dict of ", len(self.grid_dict), " and this uses quite some memory. If you are sure this not an issue please ignore")


        if self.precompute_grid:
            #if we don't have probabilities for this size of frame, we add it
            size=(frame.width, frame.height)
            if size not in self.grid_dict:
                print("adding CreateRaysModule grid for ", size)
                self.grid_dict[size]= self.compute_grid(frame)
            points_2D=self.grid_dict[size]
        else:
            #compute the grid
            points_2D=self.compute_grid(frame)


        #get 2d points
        selected_points_2D=points_2D
        if rand_indices!=None:
            selected_points_2D=torch.index_select( points_2D, dim=0, index=rand_indices) 



        #create points in 3D
        K_inv=torch.from_numpy( np.linalg.inv(frame.K) ).to("cuda").float()
        #get from screen to cam coords
        pixels_selected_screen_coords_t=selected_points_2D.transpose(0,1) #3xN
        pixels_selected_cam_coords=torch.matmul(K_inv,pixels_selected_screen_coords_t).transpose(0,1)

        nr_rays=pixels_selected_cam_coords.shape[0]
        pixels_selected_cam_coords=pixels_selected_cam_coords.view(nr_rays, 3)


        #get from cam_coords to world_coords
        tf_world_cam=frame.tf_cam_world.inverse()
        R=torch.from_numpy( tf_world_cam.linear().copy() ).to("cuda").float()
        t=torch.from_numpy( tf_world_cam.translation().copy() ).to("cuda").view(1,3).float()
        pixels_selected_world_coords=torch.matmul(R, pixels_selected_cam_coords.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t
        #get direction
        ray_dirs = pixels_selected_world_coords-t
        ray_dirs=F.normalize(ray_dirs, p=2, dim=1)

   
        #ray_origins
        ray_origins=t.repeat(nr_rays,1)

        

        return ray_origins, ray_dirs