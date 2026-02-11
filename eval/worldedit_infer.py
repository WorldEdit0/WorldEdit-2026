#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from typing import Dict, Any, List, Tuple
import sys
import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_test_items(test_json_path: str) -> List[Dict[str, Any]]:
    with open(test_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("test.json must be a LIST of dict items.")
    return data


def build_paths(item: Dict[str, Any], images_root: str) -> Tuple[str, str]:

    rel = item["image"]
    ori_path = os.path.join(images_root, rel)
    out_name = os.path.basename(rel)
    return ori_path, out_name


def build_model_and_inferencer(
    model_path: str,
    offload_dir: str,
    cuda_visible_devices: str,
    max_memory: str,
    dtype: str,
) -> InterleaveInferencer:
    # CUDA device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # dtype
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # device map
    max_memory_map = {0: max_memory}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory_map,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # load weights (ema)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch_dtype,
        force_hooks=True,
        offload_folder=offload_dir,
    )
    model = model.eval()
    print("Model loaded")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def main():
    parser = argparse.ArgumentParser("Batch infer for WorldEdit test set")

    # paths
    parser.add_argument("--model_path", type=str, required=True, help="Weights folder, contains llm_config.json/vit_config.json/ae.safetensors/ema/")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test.json (list of dict).")
    parser.add_argument("--images_root", type=str, required=True, help="Root dir that contains item['image'] relative paths.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save edited images (basename matched).")
    parser.add_argument("--save_json", type=str, default="", help="Optional: save a results json with paths/status.")

    # runtime
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max_memory", type=str, default="80GiB", help='e.g. "80GiB"')
    parser.add_argument("--offload_dir", type=str, default="/tmp/offload", help="Accelerate offload folder.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip if output image already exists.")
    parser.add_argument("--limit", type=int, default=-1, help="For debugging: only run first N items (-1 means all).")

    # inference hyper
    parser.add_argument("--think", action="store_true", help="Use think=True in inferencer call.")
    parser.add_argument("--max_think_token_n", type=int, default=1000)
    parser.add_argument("--do_sample", action="store_true", help="If set, do_sample=True else False.")
    parser.add_argument("--cfg_text_scale", type=float, default=5.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--cfg_interval", type=str, default="0.0,1.0", help='Two floats "a,b"')
    parser.add_argument("--timestep_shift", type=float, default=0.5)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    a, b = args.cfg_interval.split(",")
    cfg_interval = [float(a), float(b)]

    inferencer = build_model_and_inferencer(
        model_path=args.model_path,
        offload_dir=args.offload_dir,
        cuda_visible_devices=args.cuda_visible_devices,
        max_memory=args.max_memory,
        dtype=args.dtype,
    )

    items = load_test_items(args.test_json)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    inference_hyper = dict(
        max_think_token_n=args.max_think_token_n,
        do_sample=bool(args.do_sample),
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=cfg_interval,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )

    results: List[Dict[str, Any]] = []
    ok = 0
    skip = 0
    fail = 0

    for item in items:
        try:
            ori_path, out_name = build_paths(item, args.images_root)
            out_path = os.path.join(args.output_dir, out_name)

            if args.resume and os.path.exists(out_path):
                skip += 1
                results.append({"id": item.get("id"), "ori": ori_path, "out": out_path, "status": "skipped"})
                continue

            if not os.path.exists(ori_path):
                fail += 1
                results.append({"id": item.get("id"), "ori": ori_path, "out": out_path, "status": "missing_ori"})
                continue

            image = Image.open(ori_path).convert("RGB")

            # prompt: use implicit_instructions 
            prompt = item.get("implicit_instructions", "")

            out = inferencer(image=image, text=prompt, think=bool(args.think), **inference_hyper)
            out_img = out["image"]
            out_img.save(out_path)

            ok += 1
            results.append({"id": item.get("id"), "ori": ori_path, "out": out_path, "status": "ok"})
        except Exception as e:
            fail += 1
            results.append({"id": item.get("id"), "status": "error", "error": repr(e)})

    print(f"Done. ok={ok}, skipped={skip}, failed={fail}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved log to: {args.save_json}")


if __name__ == "__main__":
    main()
