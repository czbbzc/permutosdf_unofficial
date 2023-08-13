import torch
from typing import Optional
import math
import numpy as np
import torch.nn.functional as F

# from instant_ngp_2_py.utils.aabb import *
# from permuto_sdf  import PermutoSDF
# from permuto_sdf  import RaySamplesPacked
# from permuto_sdf  import VolumeRendering
from py_permuto_sdf import VolumeRendering
from skimage import measure
# from easypbr  import *

def sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius, sphere_center, distance_scale=1.0):
	# points=torch.cat([surface_points, offsurface_points], 0)
	# sdf=torch.cat([surface_sdf, offsurface_sdf],0)
	# all_gradients=torch.cat([surface_sdf_gradients, offsurface_sdf_gradients],0)
	points=torch.cat([offsurface_points], 0)
	sdf=torch.cat([offsurface_sdf],0)
	all_gradients=torch.cat([offsurface_sdf_gradients],0)
 


	points_in_sphere_coord=points-torch.as_tensor(sphere_center).to('cuda')
	point_dist_to_center=points_in_sphere_coord.norm(dim=-1, keepdim=True)
	# print("point_dist_to_center", point_dist_to_center)
	dists=(point_dist_to_center  - sphere_radius)*distance_scale
	# print("dists", dists)
	# print("sdf", sdf)

	loss_dists= ((sdf-dists)**2).mean()
	eikonal_loss = (all_gradients.norm(dim=-1) - distance_scale  ) **2
	loss=  loss_dists*3e3 + eikonal_loss.mean()*5e1

	#return also the loss sdf and loss eik
	loss_sdf=loss_dists
	loss_eik=eikonal_loss.mean()

	return loss, loss_sdf, loss_eik

#same as sdf_loss_sphere but takes a list of spheres as input
def sdf_loss_spheres(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_list, distance_scale=1.0):
	# points=torch.cat([surface_points, offsurface_points], 0)
	# sdf=torch.cat([surface_sdf, offsurface_sdf],0)
	# all_gradients=torch.cat([surface_sdf_gradients, offsurface_sdf_gradients],0)
	points=torch.cat([offsurface_points], 0)
	sdf=torch.cat([offsurface_sdf],0)
	all_gradients=torch.cat([offsurface_sdf_gradients],0)

	for i in range(len(sphere_list)):
		sphere=sphere_list[i]
		sphere_center=sphere.sphere_center
		sphere_radius=sphere.sphere_radius
		points_in_sphere_coord=points-torch.as_tensor(sphere_center)
		point_dist_to_center=points_in_sphere_coord.norm(dim=-1, keepdim=True)
		if i==0:
			dists=(point_dist_to_center  - sphere_radius)*distance_scale
		else: #combine sdfs by min()
			dist_to_cur_sphere=(point_dist_to_center  - sphere_radius)*distance_scale
			dists=torch.min(dists,dist_to_cur_sphere)

	loss_dists= ((sdf-dists)**2).mean()
	# eikonal_loss = torch.abs(all_gradients.norm(dim=-1) - distance_scale)
	eikonal_loss = (all_gradients.norm(dim=-1) - distance_scale  ) **2
	loss=  loss_dists*3e3 + eikonal_loss.mean()*5e1

	#return also the loss sdf and loss eik
	loss_sdf=loss_dists
	loss_eik=eikonal_loss.mean()

	return loss, loss_sdf, loss_eik

def importance_sampling_sdf_model(model_sdf, ray_samples_packed, ray_origins, ray_dirs, ray_t_exit, iter_nr_for_anneal):
	#importance sampling for the SDF model
	inv_s_imp_sampling=512
	inv_s_multiplier=1.0
	#
	sdf_sampled_packed, _=model_sdf(ray_samples_packed.samples_pos, iter_nr_for_anneal)
	ray_samples_packed.set_sdf(sdf_sampled_packed) ##set sdf
	alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
	alpha=alpha.clip(0.0, 1.0)
	transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha + 1e-7)
	weights = alpha * transmittance
	weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
	# weight_sum_per_sample[weight_sum_per_sample==0]=1e-6 #prevent nans
	weight_sum_per_sample=torch.clamp(weight_sum_per_sample, min=1e-6 )
	weights/=weight_sum_per_sample #normalize so that cdf sums up to 1
	cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
	# print("cdf min max is ", cdf.min(), cdf.max())
	ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model_sdf.training)
	sdf_sampled_packed_imp, _, =model_sdf(ray_samples_packed_imp.samples_pos, iter_nr_for_anneal)
	ray_samples_packed_imp.set_sdf(sdf_sampled_packed_imp) ##set sdf
	ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_t_exit, ray_samples_packed, ray_samples_packed_imp)
	ray_samples_packed=ray_samples_combined#swap
	ray_samples_packed=ray_samples_packed.compact_to_valid_samples() #still need to get the valid ones because we have less samples than allocated
	####SECOND ITER
	inv_s_multiplier=2.0
	sdf_sampled_packed=ray_samples_packed.samples_sdf #we already combined them and have the sdf
	alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
	alpha=alpha.clip(0.0, 1.0)
	transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha + 1e-7)
	weights = alpha * transmittance
	weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
	weight_sum_per_sample=torch.clamp(weight_sum_per_sample, min=1e-6 )
	weights/=weight_sum_per_sample #normalize so that cdf sums up to 1
	cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
	ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model_sdf.training)
	ray_samples_packed.remove_sdf() #we fuse this with ray_samples_packed_imp but we don't care about fusing the sdf
	ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_t_exit, ray_samples_packed, ray_samples_packed_imp)
	ray_samples_packed=ray_samples_combined#swap
	ray_samples_packed=ray_samples_packed.compact_to_valid_samples() #still need to get the valid ones because we have less samples than allocated

	return ray_samples_packed