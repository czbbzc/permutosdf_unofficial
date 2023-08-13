import os
import torch
from typing import Optional
import math
import numpy as np
from collections import namedtuple
import torch.nn.functional as F



from permuto.utils.nerf_utils import get_midpoint_of_sections
from permuto.utils.sdf_utils import sdf_loss_spheres
from permuto.utils.sdf_utils import sdf_loss_sphere

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True
    
def init_losses():
    loss=0.0
    loss_rgb=0.0
    loss_eikonal=0.0
    loss_curvature=0.0
    loss_lipshitz=0.0

    return loss, loss_rgb, loss_eikonal, loss_curvature, loss_lipshitz



def rgb_loss(gt_rgb, pred_rgb, does_ray_intersect_primitive):
    loss_rgb_l1= ((gt_rgb - pred_rgb).abs()*does_ray_intersect_primitive*1.0 ).mean()
    # epsilon_charbonier=0.001
    # loss_rgb_charbonier= (  torch.sqrt((gt_rgb - pred_rgb)**2 + epsilon_charbonier*epsilon_charbonier)   *does_ray_intersect_primitive*1.0 ) #Charbonnier loss from mipnerf360, acts like l2 when error is small and like l1 when error is large
    return loss_rgb_l1.mean()

def eikonal_loss(sdf_gradients):
    gradient_error = (torch.linalg.norm(sdf_gradients.reshape(-1, 3), ord=2, dim=-1) - 1.0) ** 2
    return gradient_error.mean()

def loss_sphere_init(dataset_name, nr_points, aabb, model,  iter_nr_for_anneal ):
    offsurface_points=aabb.rand_points_inside(nr_points=nr_points)
    offsurface_sdf, offsurface_sdf_gradients, feat = model.get_sdf_and_gradient(offsurface_points, iter_nr_for_anneal)
    #for phenorob
    
    # if dataset_name=="phenorobcp1":
    #     sphere_ground=SpherePy(radius=2.0, center=[0,-2.4,0])
    #     sphere_plant=SpherePy(radius=0.15, center=[0,0,0])
    #     spheres=[sphere_ground, sphere_plant]
    #     # spheres=[sphere_ground]
    #     loss, loss_sdf, gradient_error=sdf_loss_spheres(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, spheres, distance_scale=1.0)
    # elif dataset_name=="bmvs":
    #     loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    # elif dataset_name=="dtu":
    #     loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    # elif dataset_name=="easypbr":
    #     loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    # elif dataset_name=="multiface":
    #     loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    # else:
    #     # print("Using default sphere loss")
    #     loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    #     # print("dataset not known")
    #     # exit()
        
    loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)

    return loss, loss_sdf, gradient_error


def get_iter_for_anneal(iter_nr, nr_iter_sphere_fit):
    if iter_nr<nr_iter_sphere_fit:
        #####DO NOT DO THIS , it sets the iterfor anneal so high that is triggers the decay of the lipthiz loss to 0
        # iter_nr_for_anneal=999999 #dont do any c2f when fittign sphere 
        iter_nr_for_anneal=iter_nr 
    else:
        iter_nr_for_anneal=iter_nr-nr_iter_sphere_fit

    return iter_nr_for_anneal