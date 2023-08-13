import torch
from torch.nn import functional as F
from typing import Optional
import math
import numpy as np

# from permuto_sdf  import RaySampler
from py_permuto_sdf import RaySampler

#from a nr_rays x nr_samples tensor of z values, return a new tensor of some z_vals in the middle of each section. Based on Neus paper
def get_midpoint_of_sections(z_vals):
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	z_vals_except_last=z_vals[..., :-1]
	mid_z_vals = z_vals_except_last + dists * 0.5
	#now mid_z_vals is of shape nr_rays x (nr_samples -1)
	#we add another point very close to the last one just we have the same number of samples, so the last section will actually have two samples very close to each other in the middle
	mid_z_vals_last=mid_z_vals[...,-1:]
	mid_z_vals=torch.cat([mid_z_vals, mid_z_vals_last+1e-6],-1)


	# #attempt 2
	# sample_dist=1e-6 #weird, maybe just set this to something very tiny
	# dists = z_vals[..., 1:] - z_vals[..., :-1]
	# dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
	# mid_z_vals = z_vals + dists * 0.5

	return mid_z_vals

def create_samples(args, hyperparams, ray_origins, ray_dirs, jitter_samples, occupancy_grid, bounding_primitive):
	ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=bounding_primitive.ray_intersection(ray_origins, ray_dirs)

	#foreground samples
	if hyperparams.use_occupancy_grid and occupancy_grid is not None:
		fg_ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, jitter_samples)
		fg_ray_samples_packed=fg_ray_samples_packed.compact_to_valid_samples()

		


	else:
	
		fg_ray_samples_packed= RaySampler.compute_samples_fg(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, bounding_primitive.m_radius, bounding_primitive.m_center_tensor, jitter_samples)
		fg_ray_samples_packed=fg_ray_samples_packed.compact_to_valid_samples()


	#create ray samples for bg
	if not args.with_mask:
    # if not args:
		bg_ray_samples_packed= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, hyperparams.nr_samples_bg, bounding_primitive.m_radius, bounding_primitive.m_center_tensor, jitter_samples, False)
	else:
		bg_ray_samples_packed=None


	return fg_ray_samples_packed, bg_ray_samples_packed
