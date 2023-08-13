"""
Permuto configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig

from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)

from permuto.permutosdf_model import PermutoSDFModelConfig
from permuto.permutosdf_field import PermutoFieldConfig
from permuto.permutosdf_pipeline import PermutoPipelineConfig
from permuto.optimizers.optimizers import AdamWOptimizerConfig


permuto_sdf_method = MethodSpecification(
    config=TrainerConfig(
    method_name="permutosdf_f",
    steps_per_eval_image=10000,
    steps_per_eval_batch=10000,
    steps_per_save=20000,
    steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
    max_num_iterations=200001,
    mixed_precision=False,
    pipeline=PermutoPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[SDFDataset],
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=512,
            eval_num_rays_per_batch=512,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=PermutoSDFModelConfig(
            # proposal network allows for significantly smaller sdf/color network
            sdf_field=PermutoFieldConfig(),
            background_model="none",
            eval_num_rays_per_chunk=512,
        ),
    ),
    optimizers={
        # "proposal_networks": {
        #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        #     "scheduler": MultiStepSchedulerConfig(max_steps=20001, milestones=(10000, 150000, 18000)),
        # },
        "fields": {
            "optimizer": AdamWOptimizerConfig(eps=1e-15, weight_decay=0.0, lr=1e-3),
            "scheduler": MultiStepSchedulerConfig(max_steps=200001, milestones=(100000, 150000, 180000, 190000), gamma=0.3),
        },
        # "field_background": {
        #     "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
        #     "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
        # },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
    ),
    description="sdf config for Permuto",
)
