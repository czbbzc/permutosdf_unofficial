# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of Base surface model.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type, cast
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    ScaleAndShiftInvariantLoss,
    monosdf_normal_loss,
)
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider, SphereCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import normalized_depth_scale_and_shift
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler

from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

from permuto.permutosdf_field import PermutoFieldConfig

from py_permuto_sdf import Sphere

from permuto.utils.permuto_sdf_utils import init_losses
from permuto.utils.permuto_sdf_utils import get_iter_for_anneal
from permuto.utils.permuto_sdf_utils import loss_sphere_init
from permuto.utils.permuto_sdf_utils import rgb_loss
from permuto.utils.permuto_sdf_utils import eikonal_loss
from permuto.utils.permuto_sdf_utils import module_exists

from permuto.utils.common_utils import map_range_val


@dataclass
class PermutoSDFModelConfig(ModelConfig):
    """Surface Model Config"""

    _target: Type = field(default_factory=lambda: PermutoSDFModel)
    
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.05
    """Monocular normal consistency loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    # sdf_field: SDFFieldConfig = SDFFieldConfig()
    sdf_field: PermutoFieldConfig = PermutoFieldConfig()
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for background"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss multiplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""


class PermutoSDFModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: PermutoSDFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # self.scene_contraction = SceneContraction(order=float("inf"))

        # Can we also use contraction for sdf?
        # Fields
        self.field = self.config.sdf_field.setup(
            # aabb=self.scene_box.aabb,
            aabb=Sphere(0.5, torch.tensor([0, 0, 0], dtype=torch.float, device='cuda')),
            # spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            # use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        
        # sampler
        # self.sampler = NeuSSampler(
        #     num_samples=self.config.num_samples,
        #     num_samples_importance=self.config.num_samples_importance,
        #     num_samples_outside=self.config.num_samples_outside,
        #     num_upsample_steps=self.config.num_up_sample_steps,
        #     base_variance=self.config.base_variance,
        # )
        
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_samples)

        self.anneal_end = 50000

        # Collider
        self.collider = AABBBoxCollider(self.scene_box, near_plane=0.0)
        # self.collider = SphereCollider(torch.Tensor([0, 0, 0]), 0.5)
        # self.collider = None

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
            print('change to Nerf')

        # background model
        # if self.config.background_model == "grid":
        #     self.field_background = NerfactoField(
        #         self.scene_box.aabb,
        #         spatial_distortion=self.scene_contraction,
        #         num_images=self.num_train_data,
        #         use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        #     )
        # elif self.config.background_model == "mlp":
        #     position_encoding = NeRFEncoding(
        #         in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
        #     )
        #     direction_encoding = NeRFEncoding(
        #         in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        #     )

        #     self.field_background = NeRFField(
        #         position_encoding=position_encoding,
        #         direction_encoding=direction_encoding,
        #         spatial_distortion=self.scene_contraction,
        #     )
        # else:
        #     # dummy background model
        #     self.field_background = Parameter(torch.ones(1), requires_grad=False)

        # self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()

        # losses
        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups["fields"] = list(self.field.parameters())
        param_groups["fields"] = list(self.field.params)
        
        # print('############### param_groups')
        # print(param_groups)
        # param_groups["fields_pa"] = list(self.field.params)
        # param_groups["field_background"] = (
        #     [self.field_background]
        #     if isinstance(self.field_background, Parameter)
        #     else list(self.field_background.parameters())
        # )
        return param_groups
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal for cos in NeuS
        # if self.anneal_end > 0:

        #     def set_anneal(step):
        #         anneal = min([1.0, step / self.anneal_end])
        #         self.field.set_cos_anneal_ratio(anneal)

        #     callbacks.append(
        #         TrainingCallback(
        #             where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
        #             update_every_num_iters=1,
        #             func=set_anneal,
        #         )
        #     )

        return callbacks

    # @abstractmethod
    # def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict[str, Any]:
    #     """Takes in a Ray Bundle and returns a dictionary of samples and field output.

    #     Args:
    #         ray_bundle: Input bundle of rays. This raybundle should have all the
    #         needed information to compute the outputs.

    #     Returns:
    #         Outputs of model. (ie. rendered colors)
    #     """
    
    def sample_and_forward_field(self, step, batch, ray_bundle: RayBundle):
        # ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        ray_samples = self.sampler_uniform(ray_bundle)
        field_outputs, loss_outputs = self.field(step, batch, ray_samples, return_alphas=True)
        # weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
        #     field_outputs[FieldHeadNames.ALPHA]
        # )
        # bg_transmittance = transmittance[:, -1, :]

        # samples_and_field_outputs = {
        #     "ray_samples": ray_samples,
        #     "field_outputs": field_outputs,
        #     "weights": weights,
        #     "bg_transmittance": bg_transmittance,
        # }
        
        samples_and_field_outputs = field_outputs
        
        # print('############################################## 1111111111111111111')
        # for key, value in samples_and_field_outputs.items():
        #     print(key)
        #     print(value)
        
        samples_and_field_outputs["ray_samples"] = ray_samples
        
        return samples_and_field_outputs, loss_outputs

    def get_outputs(self, step, batch, ray_bundle: RayBundle):
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        assert (
            ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"

        samples_and_field_outputs, loss_outputs = self.sample_and_forward_field(step, batch, ray_bundle=ray_bundle)

        outputs = samples_and_field_outputs
        
        
        # shortcuts
        # field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
        #     Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        # )
        ray_samples = samples_and_field_outputs["ray_samples"]
        
        if step > self.field.config.nr_iter_sphere_fit:
            
            # if step > 198:
                
            #     print('yesyesyes###############')
            #     print(step)
            #     print(samples_and_field_outputs["weights_sum"].shape)
        
            weights = samples_and_field_outputs["weights"]
            # bg_transmittance = samples_and_field_outputs["bg_transmittance"]

            rgb = samples_and_field_outputs["pred_rgb"]
            # depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            # the rendered depth is point-to-point distance and we should convert to depth
            # depth = depth / ray_bundle.metadata["directions_norm"]
            normal = samples_and_field_outputs["pred_normals"]
            # normal = self.renderer_normal(semantics=samples_and_field_outputs["pred_normals"], weights=weights)
            accumulation = self.renderer_accumulation(weights=weights)
            
            # curvature_loss = samples_and_field_outputs["curvature"]

            # background model
            # if self.config.background_model != "none":
            #     assert isinstance(self.field_background, torch.nn.Module), "field_background should be a module"
            #     assert ray_bundle.fars is not None, "fars is required in ray_bundle"
            #     # sample inversely from far to 1000 and points and forward the bg model
            #     ray_bundle.nears = ray_bundle.fars
            #     assert ray_bundle.fars is not None
            #     ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            #     ray_samples_bg = self.sampler_bg(ray_bundle)
            #     # use the same background model for both density field and occupancy field
            #     assert not isinstance(self.field_background, Parameter)
            #     field_outputs_bg = self.field_background(ray_samples_bg)
            #     weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            #     rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            #     depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            #     accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            #     # merge background color to foregound color
            #     rgb = rgb + bg_transmittance * rgb_bg

            #     bg_outputs = {
            #         "bg_rgb": rgb_bg,
            #         "bg_accumulation": accumulation_bg,
            #         "bg_depth": depth_bg,
            #         "bg_weights": weights_bg,
            #     }
            # else:
            #     bg_outputs = {}

            other_outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "ray_samples": ray_samples,
                # "depth": depth,
                "normal": normal,
                # "weights": weights,
                # used to scale z_vals for free space and sdf loss
                "directions_norm": ray_bundle.metadata["directions_norm"],
                
            }
            outputs.update(other_outputs)

            # if self.training:
            #     grad_points = samples_and_field_outputs["sdf_gradients"]
            #     outputs.update({"eik_grad": grad_points})
            #     outputs.update(samples_and_field_outputs)

            if "weights_list" in samples_and_field_outputs:
                weights_list = cast(List[torch.Tensor], samples_and_field_outputs["weights_list"])
                ray_samples_list = cast(List[torch.Tensor], samples_and_field_outputs["ray_samples_list"])

                for i in range(len(weights_list) - 1):
                    outputs[f"prop_depth_{i}"] = self.renderer_depth(
                        weights=weights_list[i], ray_samples=ray_samples_list[i]
                    )
            # this is used only in viewer
            outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs, loss_outputs
    
    def get_loss_dict(self, step, batch, outputs, loss_outputs, metrics_dict=None):
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        
        loss_dict = {}
        if self.training:
            loss_dict = loss_outputs
        else:
            
            if step > self.field.config.nr_iter_sphere_fit:
                # loss_dict = loss_outputs
                # loss_dict["loss_rgb"] = loss_outputs["loss_rgb"]
                image = batch["image"].to('cuda')
                pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
                    pred_image=outputs["pred_rgb"],
                    pred_accumulation=outputs["accumulation"],
                    gt_image=image,
                )
                loss_dict["rgb_loss_eval"] = self.rgb_loss(image, pred_image)
                
            else:
                loss_dict={}

        return loss_dict

    def get_metrics_dict(self, step, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        if step > self.field.config.nr_iter_sphere_fit:
            # image = batch["image"].to(self.device)
            image = batch["image"].to('cuda')
            
            # print(image)
            # print("5555555555555555555555555555555555555555555555")
            # print(outputs["pred_rgb"])
            
            image = self.renderer_rgb.blend_background(image)
            
            # print('##############3 psnr$$$$$$$$$$$$$$$$$$$$4')
            # print(image.shape)
            # print(outputs["pred_rgb"].shape)
            
            metrics_dict["psnr"] = self.psnr(outputs["pred_rgb"], image)
        # metrics_dict["ssim"] = self.ssim(outputs["rgb"], image)
        
        # if self.training:
        #     # training statics
        #     metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
        #     metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()
        
        return metrics_dict
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, step, batch, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        outputs1={}
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs_all, _ = self.forward(step, batch, ray_bundle=ray_bundle)
            
            outputs1["pred_rgb"] = outputs_all["pred_rgb"]
            outputs1["pred_normals"] = outputs_all["pred_normals"]
            # outputs1["accumulation"] = outputs_all["accumulation"]
            # del outputs['pred_rgb']
            
            for output_name, output in outputs1.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        Args:
            outputs: Outputs of the model.
            batch: Batch of data.

        Returns:
            A dictionary of metrics.
        """
        image = batch["image"].to(self.device)
        rgb = outputs["pred_rgb"]
        # acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["pred_normals"]
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # if "depth" in batch:
        #     depth_gt = batch["depth"].to(self.device)
        #     depth_pred = outputs["depth"]

        #     # align to predicted depth and normalize
        #     scale, shift = normalized_depth_scale_and_shift(
        #         depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
        #     )
        #     depth_pred = depth_pred * scale + shift

        #     combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
        #     combined_depth = colormaps.apply_depth_colormap(combined_depth)
        # else:
        #     depth = colormaps.apply_depth_colormap(
        #         outputs["depth"],
        #         accumulation=outputs["accumulation"],
        #     )
        #     combined_depth = torch.cat([depth], dim=1)

        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            # "accumulation": combined_acc,
            # "depth": combined_depth,
            "normal": combined_normal,
        }

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        # lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict
    
    
    
    def forward(self, step, batch, ray_bundle: RayBundle):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(step, batch, ray_bundle)