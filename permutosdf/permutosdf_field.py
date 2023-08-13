"""
Field for permutosdf  model
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig

import permutohedral_encoding as permuto_enc

try:
    import tinycudann as tcnn
except ModuleNotFoundError:
    # tinycudann module doesn't exist
    pass

from permuto.models.models import SDF, RGB, NerfHash, Colorcal

# from permuto.utils.permuto_sdf_utils import get_frames_cropped
from permuto.utils.permuto_sdf_utils import init_losses
from permuto.utils.permuto_sdf_utils import get_iter_for_anneal
from permuto.utils.permuto_sdf_utils import loss_sphere_init
from permuto.utils.permuto_sdf_utils import rgb_loss
from permuto.utils.permuto_sdf_utils import eikonal_loss
from permuto.utils.permuto_sdf_utils import module_exists

from permuto.utils.common_utils import map_range_val

from permuto.utils.sdf_utils import importance_sampling_sdf_model
from permuto.utils.nerf_utils import create_samples

from permuto.schedulers.multisteplr import MultiStepLR
from permuto.schedulers.warmup import GradualWarmupScheduler

from py_permuto_sdf import OccupancyGrid

if module_exists("apex"):
    import apex
    has_apex=True
else:
    has_apex=False


@dataclass
class PermutoFieldConfig(FieldConfig):
    """Permuto Field Config"""
    
    _target: Type = field(default_factory=lambda: PermutoField)
    
    s_mult=1.0
    """multiplier for the scheduler"""
    use_color_calibration: bool = False
    """whether to use color calibration"""
    use_occupancy_grid: bool = True
    """whether to ues occupancy grid"""
    sdf_in_channels: int = 3
    """number of input channels"""
    rgb_in_channels: int = 3
    """number of input channels"""
    bg_in_channels: int = 4
    """number of input channels"""
    sdf_geom_feat_size: int = 32
    """dimension of geometric feature"""
    sdf_nr_iters_for_c2f = 10000*s_mult
    """number of sdf c2f iteration"""
    rgb_nr_iters_for_c2f: int = 1
    """number of rgb c2f iteration"""
    background_nr_iters_for_c2f: int = 1
    """number of bg c2f iteration"""
    lr: float = 1e-3
    """learning rate"""
    lr_milestones=[100000*s_mult,150000*s_mult,180000*s_mult,190000*s_mult]
    
    nr_iter_sphere_fit=4000*s_mult
    """nr iters for training with sphere SDF"""
    forced_variance_finish_iter=35000*s_mult
    """nr iters until the SDF to density transform is sharp"""  
    forced_variance_finish=0.8
    
    with_mask: bool = False
    """whether to use the mask"""
    nr_rays = 512
    """number of the rays"""
    
    eikonal_weight=0.05
    """weight of eikonal loss"""
    curvature_weight=1.5
    """weight of curvature loss"""
    lipshitz_weight=1e-5
    """weight of lipshitz loss"""
    mask_weight=0.1
    """weight of mask loss"""
    offsurface_weight=1e-4
    """weight of offsurface loss"""
    iter_start_reduce_curv=100000*s_mult
    """nr iters when we reduce the curvature loss"""
    iter_finish_reduce_curv=iter_start_reduce_curv+1001
    """nr iters when we finish reducing the curvature loss"""
    target_nr_of_samples=512*(64+16+16)
    """#the nr of rays are dynamically changed so that we use this nr of samples in a forward pass"""
    
    nr_samples_bg=32
    min_dist_between_samples=0.0001
    max_nr_samples_per_ray=64 #for the foreground
    nr_samples_imp_sampling=16
    do_importance_sampling=True  



class PermutoField(Field):
    """
    A field for Signed Distance Functions (SDF).

    Args:
        config: The configuration for the SDF field.
        aabb: An axis-aligned bounding box for the SDF field.
        num_images: The number of images for embedding appearance.
        use_average_appearance_embedding: Whether to use average appearance embedding. Defaults to False.
        spatial_distortion: The spatial distortion. Defaults to None.
    """

    config: PermutoFieldConfig

    def __init__(
        self,
        config: PermutoFieldConfig,
        # aabb: Float[Tensor, "2 3"],
        aabb,
        num_images: int,
        # use_average_appearance_embedding: bool = False,
        # spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        
        self.config = config
        
        self.aabb = aabb
        self.num_images = num_images
        
        self.model_sdf=SDF(in_channels=self.config.sdf_in_channels, boundary_primitive=self.aabb, geom_feat_size_out=self.config.sdf_geom_feat_size, nr_iters_for_c2f=self.config.sdf_nr_iters_for_c2f).to("cuda")
        self.model_rgb=RGB(in_channels=self.config.rgb_in_channels, boundary_primitive=self.aabb, geom_feat_size_in=self.config.sdf_geom_feat_size, nr_iters_for_c2f=self.config.rgb_nr_iters_for_c2f).to("cuda")
        self.model_bg=NerfHash(in_channels=self.config.bg_in_channels, boundary_primitive=self.aabb, nr_iters_for_c2f=self.config.background_nr_iters_for_c2f ).to("cuda") 
        
        if self.config.use_color_calibration:
            self.model_colorcal=Colorcal(self.num_images, 0)
            # model_colorcal=Colorcal(tensor_reel.rgb_reel.shape[0], 0)
        else:
            self.model_colorcal=None
            
        if self.config.use_occupancy_grid:
            # occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
            self.occupancy_grid = OccupancyGrid(256, 1.0, torch.tensor([0, 0, 0], dtype=torch.float, device='cuda'))
        else:
            self.occupancy_grid=None
            
        self.model_sdf.train(True)
        self.model_rgb.train(True)
        self.model_bg.train(True)
        
        self.params=[]
        self.params.append( {'params': self.model_sdf.parameters(), 'weight_decay': 0.0, 'lr': self.config.lr, 'name': "model_sdf"} )
        self.params.append( {'params': self.model_bg.parameters(), 'weight_decay': 0.0, 'lr': self.config.lr, 'name': "model_bg" } )
        self.params.append( {'params': self.model_rgb.parameters_only_encoding(), 'weight_decay': 0.0, 'lr': self.config.lr, 'name': "model_rgb_only_encoding"} )
        self.params.append( {'params': self.model_rgb.parameters_all_without_encoding(), 'weight_decay': 0.0, 'lr': self.config.lr, 'name': "model_rgb_all_without_encoding"} )
        if self.model_colorcal is not None:
            self.params.append( {'params': self.model_colorcal.parameters(), 'weight_decay': 1e-1, 'lr': self.config.lr, 'name': "model_colorcal" } )
        if has_apex:
            self.optimizer = apex.optimizers.FusedAdam (self.params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=self.config.lr)
        else:
            self.optimizer = torch.optim.AdamW (self.params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=self.config.lr)
        self.scheduler_lr_decay= MultiStepLR(self.optimizer, milestones=self.config.lr_milestones, gamma=0.3, verbose=False)   
   
        self.nr_rays_to_create = self.config.nr_rays

    def run_net(self, args, hyperparams, ray_origins, ray_dirs, img_indices, model_sdf, model_rgb, model_bg, model_colorcal, occupancy_grid, iter_nr_for_anneal,  cos_anneal_ratio, forced_variance):
        with torch.set_grad_enabled(False):
            ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=model_sdf.boundary_primitive.ray_intersection(ray_origins, ray_dirs)

            # TIME_START("create_samples")
            fg_ray_samples_packed, bg_ray_samples_packed = create_samples(args, hyperparams, ray_origins, ray_dirs, model_sdf.training, occupancy_grid, model_sdf.boundary_primitive)
            
            if hyperparams.do_importance_sampling and fg_ray_samples_packed.samples_pos.shape[0]!=0:
                fg_ray_samples_packed=importance_sampling_sdf_model(model_sdf, fg_ray_samples_packed, ray_origins, ray_dirs, ray_t_exit, iter_nr_for_anneal)
            # TIME_END("create_samples") #4ms in PermutoSDF

        # print("fg_ray_samples_packed.samples_pos.shape",fg_ray_samples_packed.samples_pos.shape)

        # TIME_START("render_fg")  
        
        # lam = lambda x:x if x != None else None
          
        if fg_ray_samples_packed.samples_pos.shape[0]==0: #if we actualyl have samples for this batch fo rays
            pred_rgb=torch.zeros_like(ray_origins)
            pred_normals=torch.zeros_like(ray_origins)
            sdf_gradients=torch.zeros_like(ray_origins)
            weights_sum=torch.zeros_like(ray_origins)[:,0:1]
            weights=torch.zeros_like(ray_origins)[:,0:1]
            bg_transmittance=torch.ones_like(ray_origins)[:,0:1]
        else:
            #foreground 
            #get sdf
            sdf, sdf_gradients, geom_feat=model_sdf.get_sdf_and_gradient(fg_ray_samples_packed.samples_pos, iter_nr_for_anneal)
            #get rgb
            rgb_samples = model_rgb( fg_ray_samples_packed.samples_pos, fg_ray_samples_packed.samples_dirs, sdf_gradients, geom_feat, iter_nr_for_anneal, model_colorcal, img_indices, fg_ray_samples_packed.ray_start_end_idx)
            #volumetric integration
            weights, weights_sum, bg_transmittance, inv_s = model_rgb.volume_renderer_neus.compute_weights(fg_ray_samples_packed, sdf, sdf_gradients, cos_anneal_ratio, forced_variance) #neus
            pred_rgb=model_rgb.volume_renderer_neus.integrate(fg_ray_samples_packed, rgb_samples, weights)

            #compute also normal by integrating the gradient
            grad_integrated_per_ray=model_rgb.volume_renderer_neus.integrate(fg_ray_samples_packed, sdf_gradients, weights)
            pred_normals=F.normalize(grad_integrated_per_ray, dim=1)
        # TIME_END("render_fg") #7.2ms in PermutoSDF   
    



        # print("bg_ray_samples_packed.samples_pos_4d",bg_ray_samples_packed.samples_pos_4d)

        # TIME_START("render_bg")    
        #run nerf bg
        if args.with_mask:
            pred_rgb_bg=None
        # else: #have to model the background
        elif bg_ray_samples_packed.samples_pos_4d.shape[0]!=0: #have to model the background
            #compute rgb and density
            rgb_samples_bg, density_samples_bg=model_bg( bg_ray_samples_packed.samples_pos_4d, bg_ray_samples_packed.samples_dirs, iter_nr_for_anneal, model_colorcal, img_indices, ray_start_end_idx=bg_ray_samples_packed.ray_start_end_idx) 
            #volumetric integration
            weights_bg, weight_sum_bg, _= model_bg.volume_renderer_nerf.compute_weights(bg_ray_samples_packed, density_samples_bg.view(-1,1))
            pred_rgb_bg=model_bg.volume_renderer_nerf.integrate(bg_ray_samples_packed, rgb_samples_bg, weights_bg)
            #combine
            pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
            pred_rgb = pred_rgb + pred_rgb_bg
        # TIME_END("render_bg")    




        # return pred_rgb, sdf_gradients, weights, weights_sum, fg_ray_samples_packed
        return pred_rgb, pred_rgb_bg, pred_normals, sdf_gradients, weights, weights_sum, fg_ray_samples_packed
        
        
    def get_outputs(
        self,
        step,
        batch,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ):
    # ) -> Dict[FieldHeadNames, Tensor]:
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        

        loss = 0
        
        loss, loss_rgb, loss_eikonal, loss_curvature, loss_lipshitz=init_losses() 
        
        iter_nr_for_anneal=get_iter_for_anneal(step, self.config.nr_iter_sphere_fit)
        in_process_of_sphere_init=step<self.config.nr_iter_sphere_fit
        just_finished_sphere_fit=step==self.config.nr_iter_sphere_fit
        
        outputs = {}
        loss_dict = {}
        
        if in_process_of_sphere_init:
            loss, loss_sdf, loss_eikonal= loss_sphere_init('dtu', 30000, self.aabb, self.model_sdf, iter_nr_for_anneal )
            cos_anneal_ratio=1.0
            forced_variance=0.8
            
            loss_dict["loss_sdf"] = loss_sdf*3e3
            loss_dict["loss_eikonal"] = loss_eikonal*5e1
            
        else:
            with torch.set_grad_enabled(False):
                cos_anneal_ratio=map_range_val(iter_nr_for_anneal, 0.0, self.config.forced_variance_finish_iter, 0.0, 1.0)
                forced_variance=map_range_val(iter_nr_for_anneal, 0.0, self.config.forced_variance_finish_iter, 0.3, self.config.forced_variance_finish)

                # ray_origins, ray_dirs, gt_selected, gt_mask, img_indices=PermutoSDF.random_rays_from_reel(tensor_reel, nr_rays_to_create) 
                # gt_indices = batch["indices"].to('cuda')
                
                img_indices = ray_samples.camera_indices.squeeze()[:,0]

                inputs = ray_samples.frustums.get_start_positions()

                # ray_origins = inputs.view(-1, 3)
                ray_origins = inputs[:,0,:]

                directions = ray_samples.frustums.directions
                # ray_dirs = directions.reshape(-1, 3)
                ray_dirs = directions[:,0,:]
                
                # if step > 198:
                #     print('###########3', step, '####################')
                #     print('---------------------- ray_origins')
                #     print(inputs.shape)
                #     print(inputs)
                #     print(ray_origins.shape)
                #     print(ray_origins)
                    
                #     print('---------------------- ray_dirs')
                #     print(directions.shape)
                #     print(directions)
                #     print(ray_dirs.shape)
                #     print(ray_dirs)

                # ray_origins.requires_grad_(True)
                
                ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=self.aabb.ray_intersection(ray_origins, ray_dirs)

            pred_rgb, pred_rgb_bg, pred_normals, sdf_gradients, weights, weights_sum, fg_ray_samples_packed = self.run_net(self.config, self.config, ray_origins, ray_dirs, img_indices, self.model_sdf, self.model_rgb, self.model_bg, self.model_colorcal, self.occupancy_grid, iter_nr_for_anneal,  cos_anneal_ratio, forced_variance)
                
            
            
                            
            outputs["pred_rgb"] = pred_rgb
            outputs["pred_rgb_bg"] = pred_rgb_bg
            outputs["pred_normals"] = pred_normals
            outputs["sdf_gradients"] = sdf_gradients
            outputs["weights_sum"] = weights_sum
            outputs["weights"] = weights
            # outputs["fg_ray_samples_packed"] = fg_ray_samples_packed
            
            gt_selected = batch["image"].to('cuda')
            
            # loss_rgb=rgb_loss(gt_selected, pred_rgb, does_ray_intersect_box)
            # loss+=loss_rgb
            # loss_dict["loss_rgb"] = loss_rgb
            
            if self.training:
                # print('##############3', step, '##############3')
                # print(self.training)

            
                # gt_selected = batch["image"].to(self.device)
                
                #losses -----
                
                
                #rgb loss
                
                # print('----------------------- rgb_start')
                # print(gt_selected.shape)
                # print(gt_selected)
                # print(pred_rgb.shape)
                # print(pred_rgb)
                # print('------------------------rgb finish')
                
                loss_rgb=rgb_loss(gt_selected, pred_rgb, does_ray_intersect_box)
                loss+=loss_rgb
                loss_dict["loss_rgb"] = loss_rgb
                
                #eikonal loss
                loss_eikonal =eikonal_loss(sdf_gradients)
                loss+=loss_eikonal*self.config.eikonal_weight
                loss_dict["loss_eikonal"] = loss_eikonal*self.config.eikonal_weight
                
                #curvature loss
                global_weight_curvature=map_range_val(iter_nr_for_anneal, self.config.iter_start_reduce_curv, self.config.iter_finish_reduce_curv, 1.0, 0.000) #once we are converged onto good geometry we can safely descrease it's weight so we learn also high frequency detail geometry.
                if global_weight_curvature>0.0:
                # if True:
                    sdf_shifted, sdf_curvature=self.model_sdf.get_sdf_and_curvature_1d_precomputed_gradient_normal_based( fg_ray_samples_packed.samples_pos, sdf_gradients, iter_nr_for_anneal)
                    loss_curvature=sdf_curvature.mean() 
                    loss+=loss_curvature* self.config.curvature_weight*global_weight_curvature
                    loss_dict["loss_curvature"] = loss_curvature* self.config.curvature_weight*global_weight_curvature
                    
                #loss for empty space sdf            
                if self.config.use_occupancy_grid:
                    #highsdf just to avoice voxels becoming "occcupied" due to their sdf dropping to zero
                    offsurface_points=self.model_sdf.boundary_primitive.rand_points_inside(nr_points=1024)
                    sdf_rand, _=self.model_sdf( offsurface_points, iter_nr_for_anneal)
                    loss_offsurface_high_sdf=torch.exp(-1e2 * torch.abs(sdf_rand)).mean()
                    loss+=loss_offsurface_high_sdf*self.config.offsurface_weight
                    loss_dict["loss_offsurface_high_sdf"] = loss_offsurface_high_sdf*self.config.offsurface_weight
                    
                #loss on lipshitz
                loss_lipshitz=self.model_rgb.mlp.lipshitz_bound_full()
                if iter_nr_for_anneal>=self.config.iter_start_reduce_curv:
                    loss+=loss_lipshitz.mean()*self.config.lipshitz_weight
                    loss_dict["loss_lipshitz"] = loss_lipshitz.mean()*self.config.lipshitz_weight
                    
                with torch.set_grad_enabled(False):
                    #update occupancy
                    if step%8==0 and self.config.use_occupancy_grid:
                        grid_centers_random, grid_center_indices=self.occupancy_grid.compute_random_sample_of_grid_points(256*256*4,True)
                        sdf_grid,_=self.model_sdf( grid_centers_random, iter_nr_for_anneal) 
                        self.occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf_grid, self.model_rgb.volume_renderer_neus.get_last_inv_s(), 1e-4 )
                        # occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf_grid, model_rgb.volume_renderer_neus.get_last_inv_s().item(), 1e-4 )

                    #adjust nr_rays_to_create based on how many samples we have in total
                    # cur_nr_samples=fg_ray_samples_packed.samples_pos.shape[0]
                    # # if cur_nr_samples != 0:
                    # multiplier_nr_samples=float(self.config.target_nr_of_samples)/cur_nr_samples
                    # multiplier_nr_samples = np.around(multiplier_nr_samples, 6)
                    # multiplier_nr_samples = float(multiplier_nr_samples)
                    
                    # if step > 10445:
                    #     print(step)
                    #     print(self.nr_rays_to_create)
                    #     print(multiplier_nr_samples)
                    
                    # self.nr_rays_to_create=int(self.nr_rays_to_create*multiplier_nr_samples)

                    #increase also the WD on the encoding of the model_rgb to encourage the network to get high detail using the model_sdf
                    if iter_nr_for_anneal>=self.config.iter_start_reduce_curv:
                        for group in self.optimizer.param_groups:
                            if group["name"]=="model_rgb_only_encoding":
                                group["weight_decay"]=1.0
                            #decrease eik_w as it seems to also slightly help with getting more detail on the surface
                            self.config.eikonal_weight=0.01
                            
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                # if just_finished_sphere_fit:
                #     scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=3000, after_scheduler=scheduler_lr_decay) 
                # if not in_process_of_sphere_init:
                #     scheduler_warmup.step() #this will call the scheduler for the decay
            
                        
            
            
        return outputs, loss_dict
            

            
            

        # camera_indices = ray_samples.camera_indices.squeeze()

        # inputs = ray_samples.frustums.get_start_positions()
        # inputs = inputs.view(-1, 3)

        # directions = ray_samples.frustums.directions
        # directions_flat = directions.reshape(-1, 3)

        # inputs.requires_grad_(True)
        # with torch.enable_grad():
        #     hidden_output = self.forward_geonetwork(inputs)
        #     sdf, geo_feature = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        # d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        # gradients = torch.autograd.grad(
        #     outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        # )[0]

        # rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

        # rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        # sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        # gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        # normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        # outputs.update(
        #     {
        #         FieldHeadNames.RGB: rgb,
        #         FieldHeadNames.SDF: sdf,
        #         FieldHeadNames.NORMALS: normals,
        #         FieldHeadNames.GRADIENT: gradients,
        #     }
        # )

        # if return_alphas:
        #     alphas = self.get_alpha(ray_samples, sdf, gradients)
        #     outputs.update({FieldHeadNames.ALPHA: alphas})

        # return outputs  
        
        
        
    def forward(
        self, step, batch, ray_samples: RaySamples, compute_normals: bool = False, return_alphas: bool = False
    ):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            compute normals: not currently used in this implementation.
            return_alphas: Whether to return alpha values
        """
        outputs, loss_dict = self.get_outputs(step, batch, ray_samples, return_alphas=return_alphas)
        return outputs, loss_dict