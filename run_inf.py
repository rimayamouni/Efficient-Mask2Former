import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

VAL_DIR = "/home/user/Desktop/val"
OUTPUT_DIR = "./outputs_trt"
CKPT_PATH = "/home/user/Desktop/model_0014999.pth"
TRT_ENGINE_PATH = "./mask2former_trt.pt"
MAX_IMAGES = 20
RESIZE_WIDTH = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml")
cfg.MODEL.WEIGHTS = CKPT_PATH
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.freeze()

device = torch.device(cfg.MODEL.DEVICE)
print(f"[INFO] Using device: {device}")

predictor = DefaultPredictor(cfg)
model = predictor.model.eval().to(device)

use_trt = False
use_fp16 = False
trt_model = None

if device.type == "cuda":
    try:
        import torch_tensorrt
        if os.path.exists(TRT_ENGINE_PATH):
            print("[INFO] Loading existing TensorRT engine...")
            trt_model = torch.jit.load(TRT_ENGINE_PATH).eval().to(device)
            use_trt = True
        else:
            print("[INFO] Exporting TensorRT engine for the first time...")
            dummy_input = torch.randn(1, 3, RESIZE_WIDTH, RESIZE_WIDTH, device=device)
            trt_model = torch_tensorrt.ts.compile(
                model,
                inputs=[dummy_input],
                enabled_precisions={torch.float16},
                workspace_size=1 << 25,
                truncate_long_and_double=True
            )
            torch.jit.save(trt_model, TRT_ENGINE_PATH)
            use_trt = True
        print("[INFO] TensorRT engine ready.")
    except ModuleNotFoundError:
        use_fp16 = True
        print("[INFO] torch_tensorrt not found. Using FP16 PyTorch fallback.")

def get_all_images(root_dir):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)

def preprocess_image(img):
    if RESIZE_WIDTH is not None and img.shape[1] > RESIZE_WIDTH:
        scale = RESIZE_WIDTH / img.shape[1]
        new_h = int(img.shape[0] * scale)
        img = cv2.resize(img, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def colorize_mask(mask):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
    return colors[mask % 256]

def save_mask(mask, path, colorized=True):
    Image.fromarray(mask.astype(np.uint8)).save(path)
    if colorized:
        color_mask = colorize_mask(mask)
        Image.fromarray(color_mask).save(path.replace(".png", "_color.png"))

image_list = get_all_images(VAL_DIR)
if not image_list:
    raise RuntimeError(f"No images found in {VAL_DIR}")
image_list = image_list[:MAX_IMAGES]

print(f"[INFO] Found {len(image_list)} images.")

def find_max_batch(model, device, max_try=64, fp16=False):
    print("[INFO] Finding max batch size...")
    batch_size = 1
    while batch_size <= max_try:
        try:
            dummy = torch.randn(batch_size, 3, RESIZE_WIDTH, RESIZE_WIDTH, device=device)
            if fp16:
                dummy = dummy.half()
            if use_trt:
                _ = trt_model(dummy)
            else:
                _ = model([{"image": dummy[j]} for j in range(batch_size)])
            batch_size *= 2
        except RuntimeError as e:
            torch.cuda.empty_cache()
            batch_size //= 2
            break
    batch_size = max(1, batch_size)
    print(f"[INFO] Using batch_size={batch_size}")
    return batch_size

batch_size = find_max_batch(model, device, fp16=use_fp16)

dummy = torch.zeros(1, 3, RESIZE_WIDTH, RESIZE_WIDTH, device=device)
if use_fp16:
    dummy = dummy.half()
try:
    _ = model(dummy)
except:
    pass

total_start = time.time()

with torch.no_grad():
    for i in tqdm(range(0, len(image_list), batch_size), desc="Batches", ncols=100):
        batch_paths = image_list[i:i+batch_size]
        images = []
        for path in batch_paths:
            img = cv2.imread(path)
            img = preprocess_image(img)
            images.append(torch.from_numpy(img.transpose(2,0,1)).float())

        batch_tensor = torch.stack(images).to(device)
        if use_fp16:
            batch_tensor = batch_tensor.half() / 255.0
        else:
            batch_tensor = batch_tensor / 255.0

        t0 = time.time()
        if use_trt:
            outputs = trt_model(batch_tensor)
        else:
            outputs = model([{"image": batch_tensor[j]} for j in range(batch_tensor.shape[0])])
        infer_time = time.time() - t0

        for j, path in enumerate(batch_paths):
            mask = outputs[j]["sem_seg"].argmax(dim=0).cpu().numpy()
            base_name = os.path.splitext(os.path.basename(path))[0]
            save_mask(mask, os.path.join(OUTPUT_DIR, f"{base_name}_mask.png"))

        print(f" Batch {i//batch_size + 1} | Time: {infer_time:.2f}s | Avg/image: {infer_time/len(batch_paths):.2f}s")

total_time = time.time() - total_start
avg_time = total_time / len(image_list)
print("\n Inference complete.")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Total time: {total_time:.2f}s | Avg/image: {avg_time:.2f}s")
