# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script with Global Pruning and CO2 Emission Tracking.
"""

import copy
import itertools
import logging
import os
import warnings

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.nn.utils.prune as prune

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Mask2Former
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

# CO₂ emission tracking
from eco2ai import Tracker

# Pruning amount (20%)
PRUNING_AMOUNT = 0.2

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg", "ade20k_panoptic_seg", "cityscapes_panoptic_seg", "mapillary_vistas_panoptic_seg"]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "cityscapes_instance":
            assert torch.cuda.device_count() > comm.get_rank()
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert torch.cuda.device_count() > comm.get_rank()
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert torch.cuda.device_count() > comm.get_rank()
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert torch.cuda.device_count() > comm.get_rank()
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(f"No Evaluator for the dataset {dataset_name} with type {evaluator_type}")
        return evaluator_list[0] if len(evaluator_list) == 1 else DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = {
            "mask_former_semantic": MaskFormerSemanticDatasetMapper,
            "mask_former_panoptic": MaskFormerPanopticDatasetMapper,
            "mask_former_instance": MaskFormerInstanceDatasetMapper,
            "coco_instance_lsj": COCOInstanceNewBaselineDatasetMapper,
            "coco_panoptic_lsj": COCOPanopticNewBaselineDatasetMapper,
        }.get(cfg.INPUT.DATASET_MAPPER_NAME, None)
        return build_detection_train_loader(cfg, mapper=mapper(cfg, True) if mapper else None)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )
        params, memo = [], set()
        for name, module in model.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                if not param.requires_grad or param in memo:
                    continue
                memo.add(param)
                hparams = copy.copy(defaults)
                if "backbone" in name:
                    hparams["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
                if pname in ["relative_position_bias_table", "absolute_pos_embed"]:
                    hparams["weight_decay"] = 0.0
                if isinstance(module, norm_types):
                    hparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hparams["weight_decay"] = weight_decay_embed
                params.append({"params": [param], **hparams})

        def maybe_clip(optim):
            if cfg.SOLVER.CLIP_GRADIENTS.ENABLED and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                class ClippedOptimizer(optim):
                    def step(self, closure=None):
                        torch.nn.utils.clip_grad_norm_(itertools.chain(*[g['params'] for g in self.param_groups]),
                                                       cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                        super().step(closure)
                return ClippedOptimizer
            return optim

        optim_class = torch.optim.SGD if cfg.SOLVER.OPTIMIZER == "SGD" else torch.optim.AdamW
        optimizer = maybe_clip(optim_class)(params, lr=cfg.SOLVER.BASE_LR)
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [cls.build_evaluator(cfg, name, os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
                      for name in cfg.DATASETS.TEST]
        res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})


def apply_global_pruning(model, amount=0.2):
    parameters_to_prune = [(m, 'weight') for _, m in model.named_modules()
                           if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    print(f"Applying global pruning on {len(parameters_to_prune)} layers with {amount*100:.0f}% sparsity")
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for m, _ in parameters_to_prune:
        prune.remove(m, 'weight')


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    pruning_pct = int(PRUNING_AMOUNT * 100)
    cfg.OUTPUT_DIR = os.path.join("output", f"mask2former_eco2ai_globalpruned_{pruning_pct}pct")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    # Initialize CO₂ emission tracking
    tracker = Tracker(
        project_name="Mask2Former",
        experiment_description="Training with global pruning and CO2 tracking",
        file_name=os.path.join(cfg.OUTPUT_DIR, "emission.csv")
    )
    tracker.start()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        tracker.stop()
        return res

    trainer = Trainer(cfg)
    apply_global_pruning(trainer.model, amount=PRUNING_AMOUNT)
    trainer.resume_or_load(resume=args.resume)
    results = trainer.train()
    tracker.stop()
    return results


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
