import time
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import (
    MaskFormerSemanticDatasetMapper,
    add_maskformer2_config,
)

from codecarbon import EmissionsTracker


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (
                    self.decay * self.shadow[n] + (1 - self.decay) * p.detach()
                )

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])


def build_probe_loader(cfg):
    return build_detection_train_loader(
        cfg,
        mapper=MaskFormerSemanticDatasetMapper(cfg, True),
    )


def compute_mask_class_saliency(model, data):
    outputs = model(data)
    loss_dict = outputs if isinstance(outputs, dict) else outputs["loss_dict"]
    loss = sum(loss_dict.values())
    loss.backward(retain_graph=True)

    saliency = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        fisher = (p.grad ** 2)
        mask_weighted = fisher * (p.abs() + 1e-6)
        saliency[name] = mask_weighted.detach()

    model.zero_grad()
    return saliency


def prune_conv(module, score, sparsity):
    if not isinstance(module, torch.nn.Conv2d):
        return

    importance = score.mean(dim=(1, 2, 3))
    k = int(sparsity * importance.numel())
    if k <= 0:
        return

    threshold = torch.topk(importance, k, largest=False).values.max()
    mask = importance > threshold

    module.weight.data = module.weight.data[mask]
    module.out_channels = mask.sum().item()


def prune_linear(module, score, sparsity):
    if not isinstance(module, torch.nn.Linear):
        return

    importance = score.mean(dim=1)
    k = int(sparsity * importance.numel())
    if k <= 0:
        return

    threshold = torch.topk(importance, k, largest=False).values.max()
    mask = importance > threshold

    module.weight.data = module.weight.data[mask]


def apply_pruning(model, base_sparsity, saliency):
    for name, module in model.named_modules():
        if "backbone" in name:
            continue

        if any(x in name for x in ["norm", "embedding", "pos", "head"]):
            continue

        score = saliency.get(name, None)
        if score is None:
            continue

        class_factor = score.std() + 0.5
        sparsity = min(0.28, max(0.09, base_sparsity * class_factor))

        if isinstance(module, torch.nn.Conv2d):
            prune_conv(module, score, sparsity)
        elif isinstance(module, torch.nn.Linear):
            prune_linear(module, score, sparsity)


class CarbonController:
    def __init__(self):
        self.prev = None

    def update(self, emission, base_sparsity):
        if self.prev is None:
            self.prev = emission
            return base_sparsity

        delta = emission - self.prev

        if delta > 0:
            base_sparsity = min(0.30, base_sparsity + 0.05)
        else:
            base_sparsity = max(0.05, base_sparsity - 0.02)

        self.prev = emission
        return base_sparsity


def dynamic_sparsity(iteration, max_iter):
    r = iteration / max_iter
    if r < 0.3:
        return 0.10
    elif r < 0.7:
        return 0.20
    return 0.25


class Trainer(DefaultTrainer):
    def __init__(self, cfg, tracker):
        super().__init__(cfg)

        self.ema = EMA(self.model)
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.probe_loader = iter(build_probe_loader(cfg))
        self.carbon = CarbonController()
        self.tracker = tracker

        self.step = 0

    def run_step(self):
        self.model.train()

        data = next(self._trainer._data_loader_iter)
        outputs = self.model(data)
        loss = sum(outputs.values())

        probe = next(self.probe_loader)
        saliency = compute_mask_class_saliency(self.model, probe)

        emission = self.tracker._total_emissions
        base = dynamic_sparsity(self.iter, self.max_iter)
        sparsity = self.carbon.update(emission, base)

        if self.step % 5 == 0:
            apply_pruning(self.model, sparsity, saliency)

        self.step += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ema.update(self.model)


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    setup_logger(output=cfg.OUTPUT_DIR)

    return cfg


def main(args):
    cfg = setup(args)

    tracker = EmissionsTracker(
        project_name="GreenMask2Former",
        output_dir=cfg.OUTPUT_DIR,
        output_file="emissions.csv",
    )

    tracker.start()

    trainer = Trainer(cfg, tracker)
    trainer.resume_or_load(resume=args.resume)

    start = time.time()
    trainer.train()
    end = time.time()

    emissions = tracker.stop()

    print(f"Training time: {end - start:.2f}s")
    print(f"CO2 emissions: {emissions:.6f} kg")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
