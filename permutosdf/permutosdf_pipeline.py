import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional
from pathlib import Path
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from time import time
from PIL import Image

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils import profiler

from permuto.permutosdf_model import PermutoSDFModel, PermutoSDFModelConfig

import datetime

def map_range_val( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    input_clamped=max(input_start, min(input_end, input_val))
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)


@dataclass
class PermutoPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: PermutoPipeline)
    """target class to instantiate"""
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: PermutoSDFModelConfig = PermutoSDFModelConfig()
    """specifies the model config"""


class PermutoPipeline(VanillaPipeline):
    def __init__(
        self,
        config: PermutoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(PermutoSDFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
            
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        
        # self.model.field.model_sdf.train(True)
        # self.model.field.model_rgb.train(True)
        # self.model.field.model_bg.train(True)
        
        ray_bundle, batch = self.datamanager.next_train(step)
        # start_time = datetime.datetime.now()
        model_outputs, loss_outputs = self._model(step, batch, ray_bundle)
        # finish_time = datetime.datetime.now()
        
        # if step % 1000 == 0:
        #     with open(r'train_time1.txt', 'a') as f:
        #         f.write('step: ' + str(step) + '\n')
        #         f.write('use_time: ' + str(finish_time - start_time) + '\n')
        
        metrics_dict = self.model.get_metrics_dict(step, model_outputs, batch)
        
        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(step, batch, model_outputs, loss_outputs, metrics_dict=metrics_dict)
        
        # 9500 10501
        # global_weight_curvature=map_range_val(step, self.model.config.start_reduce_curv, self.model.config.finish_reduce_curv, 1.0, 0.000)
        # if global_weight_curvature > 0.0:
        #     loss_dict["curv_loss"] = loss_dict["curv_loss"] * global_weight_curvature
        # else:
        #     del loss_dict["curv_loss"]
        
        # if step < self.model.config.start_reduce_curv:
        #     del loss_dict["loss_lipshitz"]

        if step % 1000 == 0:
            with open(r'train_time1.txt', 'a') as f:
                # f.write('loss_dict :' + '\n')
                # f.write(str(loss_dict) + '\n')
        
                f.write('metrics_dict: ' + str(metrics_dict) + '\n')
                # f.write(str(metrics_dict) + '\n')
                # f.write('\n')
                f.write('\n')
            # print('time time time finish !')
            # now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # f.write(str(now_time) + '\n')
            # print(now_time)

        # print('########### metrics_dict ############')
        # print(metrics_dict)

        # print('########### loss_dict ############')
        # print(loss_dict)
        
            
        # if (step == self.model.config.start_reduce_curv-1) or (step == self.model.config.start_reduce_curv) or (step == self.model.config.start_reduce_curv+1):
        #     print(step)
        #     print(global_weight_curvature)
        #     print(loss_dict)
        #     print(self.model.config.start_reduce_curv)
        #     print(self.model.config.finish_reduce_curv)
        #     print(self.model)
            
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        
        self.eval()
        
        # self.model.field.model_sdf.train(False)
        # self.model.field.model_rgb.train(False)
        # self.model.field.model_bg.train(False)
        
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs, loss_outputs = self.model(step, batch, ray_bundle)
        metrics_dict = self.model.get_metrics_dict(step, model_outputs, batch)
        loss_dict = self.model.get_loss_dict(step, batch, model_outputs, loss_outputs, metrics_dict=metrics_dict)
        
        self.train()
        
        # self.model.field.model_sdf.train(True)
        # self.model.field.model_rgb.train(True)
        # self.model.field.model_bg.train(True)
        
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        # outputs = self.model.get_outputs_for_camera_ray_bundle(step, batch, camera_ray_bundle)
        
        if step > self.model.field.config.nr_iter_sphere_fit:
            outputs = self.model.get_outputs_for_camera_ray_bundle(step, batch, camera_ray_bundle)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        else:
            metrics_dict = {}
            images_dict = {}
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
    
    # @profiler.time_function
    # def get_average_eval_image_metrics(
    #     self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    # ):
    #     """Iterate over all the images in the eval dataset and get the average.

    #     Args:
    #         step: current training step
    #         output_path: optional path to save rendered images to
    #         get_std: Set True if you want to return std with the mean metric.

    #     Returns:
    #         metrics_dict: dictionary of metrics
    #     """
    #     self.eval()
    #     metrics_dict_list = []
    #     assert isinstance(self.datamanager, VanillaDataManager)
    #     num_images = len(self.datamanager.fixed_indices_eval_dataloader)
    #     with Progress(
    #         TextColumn("[progress.description]{task.description}"),
    #         BarColumn(),
    #         TimeElapsedColumn(),
    #         MofNCompleteColumn(),
    #         transient=True,
    #     ) as progress:
    #         task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
    #         for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
    #             # time this the following line
    #             inner_start = time()
    #             height, width = camera_ray_bundle.shape
    #             num_rays = height * width
    #             outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    #             metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

    #             if output_path is not None:
    #                 camera_indices = camera_ray_bundle.camera_indices
    #                 assert camera_indices is not None
    #                 for key, val in images_dict.items():
    #                     Image.fromarray((val * 255).byte().cpu().numpy()).save(
    #                         output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
    #                     )
    #             assert "num_rays_per_sec" not in metrics_dict
    #             metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
    #             fps_str = "fps"
    #             assert fps_str not in metrics_dict
    #             metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
    #             metrics_dict_list.append(metrics_dict)
    #             progress.advance(task)
    #     # average the metrics list
    #     metrics_dict = {}
    #     for key in metrics_dict_list[0].keys():
    #         if get_std:
    #             key_std, key_mean = torch.std_mean(
    #                 torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
    #             )
    #             metrics_dict[key] = float(key_mean)
    #             metrics_dict[f"{key}_std"] = float(key_std)
    #         else:
    #             metrics_dict[key] = float(
    #                 torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
    #             )
    #     self.train()
    #     return metrics_dict
    
    