# run_eval.py
# -*- coding: utf-8 -*-

import argparse
import base64
import html
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from prompts import (
    VISUAL_CONSISTENCY_TEMPLATE,
    VISUAL_QUALITY_TEMPLATE,
    INSTRUCTION_FOLLOWING_TEMPLATE,
    KNOWLEDGE_PLAUSIBILITY_TEMPLATE,
)

# -----------------------------
# JSON parsing
# -----------------------------

JSON_FENCE_RE = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)


def extract_json_from_markdown(text: str):
    """Parse JSON from fenced ```json ...``` or raw JSON content."""
    if text is None:
        return None
    m = JSON_FENCE_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()
    candidate = html.unescape(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# -----------------------------
# Image to data-url (center crop)
# -----------------------------

def image_to_data_url(
    image_path: str,
    *,
    target_landscape=(1536, 1024),
    target_portrait=(1024, 1536),
) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_w, target_h = target_landscape if w >= h else target_portrait

    # upscale if needed
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        w, h = img.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -----------------------------
# OpenAI compatible client wrapper
# -----------------------------

@dataclass
class ClientConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int
    max_retries: int


class EvalClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def _call(self, messages):
        last_err = None
        for _ in range(self.cfg.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    timeout=self.cfg.timeout,
                )
                content = completion.choices[0].message.content
                return extract_json_from_markdown(content)
            except Exception as e:
                last_err = e

        return {"error": f"request_failed: {type(last_err).__name__}", "detail": str(last_err)}

    def eval_two_images(self, prompt: str, ori_path: str, edit_path: str):
        ori_url = image_to_data_url(ori_path)
        edit_url = image_to_data_url(edit_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": ori_url}},
                {"type": "image_url", "image_url": {"url": edit_url}},
            ],
        }]
        return self._call(messages)

    def eval_one_image(self, prompt: str, img_path: str):
        img_url = image_to_data_url(img_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }]
        return self._call(messages)


# -----------------------------
# Prompt builders
# -----------------------------

def prompt_visual_consistency(instruct: str) -> str:
    return VISUAL_CONSISTENCY_TEMPLATE.format(instruct=instruct)

def prompt_visual_quality() -> str:
    return VISUAL_QUALITY_TEMPLATE

def prompt_instruction_following(instruct: str) -> str:
    return INSTRUCTION_FOLLOWING_TEMPLATE.format(instruct=instruct)

def prompt_knowledge_plausibility(instruct: str, mode: str) -> str:
    return KNOWLEDGE_PLAUSIBILITY_TEMPLATE.format(instruct=instruct, mode=mode)


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Path mapping (new metadata format)
# -----------------------------

def build_paths(ori_root: str, edit_root: str, meta_item: dict) -> tuple[str, str]:
    """
    meta_item["image"] like: 3.png
    ori_path: join(ori_root, meta_item["image"])
    edit_path: join(edit_root, basename(meta_item["image"])) -> output/3.png
    """
    ori_rel = meta_item["image"]
    ori_path = os.path.join(ori_root, ori_rel)

    base = os.path.basename(ori_rel)  # "3.png"

    edit_path = os.path.join(edit_root, base)
    return ori_path, edit_path


# -----------------------------
# Evaluation per item
# -----------------------------

def eval_one_item(client: EvalClient, ori_path: str, edit_path: str, meta: dict) -> dict:
    instruct = meta.get("implicit_instructions", "")
    mode = meta.get("mode", "")

    vc = client.eval_two_images(prompt_visual_consistency(instruct), ori_path, edit_path)
    vq = client.eval_one_image(prompt_visual_quality(), edit_path)
    ins = client.eval_two_images(prompt_instruction_following(instruct), ori_path, edit_path)
    kp = client.eval_two_images(prompt_knowledge_plausibility(instruct, mode), ori_path, edit_path)

    return {
        "id": meta.get("id"),
        "ori_image": ori_path,
        "edit_image": edit_path,
        "item_data": meta,  # keep original meta for traceability
        "visual_consistency": vc,
        "visual_quality": vq,
        "instruction_following": ins,
        "knowledge_plausibility": kp,
    }


def is_missing(x) -> bool:
    return x is None or x == {} or x == ""


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Open-source friendly eval runner (new metadata format).")

    # dataset paths
    parser.add_argument("--metadata", required=True, help="Path to metadata.json (list of dicts).")
    parser.add_argument("--ori_root", required=True, help="Root directory that contains meta['image'] paths.")
    parser.add_argument("--edit_root", required=True, help="Directory that contains edited images (e.g., output/).")
    parser.add_argument("--output", required=True, help="Output JSON path (results).")

    # client config
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible base_url.")
    parser.add_argument("--model", required=True, help="Vision-language model name.")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY",
                        help="Env var name for API key (default: OPENAI_API_KEY).")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout seconds.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per request.")

    # runtime
    parser.add_argument("--workers", type=int, default=8, help="Thread workers.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output: only fill missing fields.")

    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Please set env var: {args.api_key_env}")

    meta_list = load_json(args.metadata)
    if not isinstance(meta_list, list):
        raise ValueError("metadata.json must be a LIST of items (new format).")

    cfg = ClientConfig(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    client = EvalClient(cfg)

    # load existing results if resume
    results = []
    by_edit = {}
    if args.resume and os.path.exists(args.output):
        try:
            results = load_json(args.output)
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and "edit_image" in r:
                        by_edit[r["edit_image"]] = r
            else:
                results = []
        except Exception:
            results = []

    def process(meta_item: dict):
        ori_path, edit_path = build_paths(args.ori_root, args.edit_root, meta_item)

        if not os.path.exists(ori_path) or not os.path.exists(edit_path):
            return None  # skip missing pairs

        # resume path
        if args.resume and edit_path in by_edit:
            r = by_edit[edit_path]
            changed = False

            instruct = meta_item.get("implicit_instructions", "")
            mode = meta_item.get("mode", "")

            if is_missing(r.get("visual_consistency")):
                r["visual_consistency"] = client.eval_two_images(
                    prompt_visual_consistency(instruct), ori_path, edit_path
                )
                changed = True

            if is_missing(r.get("visual_quality")):
                r["visual_quality"] = client.eval_one_image(
                    prompt_visual_quality(), edit_path
                )
                changed = True

            if is_missing(r.get("instruction_following")):
                r["instruction_following"] = client.eval_two_images(
                    prompt_instruction_following(instruct), ori_path, edit_path
                )
                changed = True

            if is_missing(r.get("knowledge_plausibility")):
                r["knowledge_plausibility"] = client.eval_two_images(
                    prompt_knowledge_plausibility(instruct, mode), ori_path, edit_path
                )
                changed = True

            # keep meta synced (in case metadata updated)
            r["id"] = meta_item.get("id")
            r["item_data"] = meta_item
            r["ori_image"] = ori_path
            r["edit_image"] = edit_path

            return r if changed else None

        # fresh eval
        return eval_one_item(client, ori_path, edit_path, meta_item)

    updated = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, m) for m in meta_list]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            out = fut.result()
            if out is None:
                continue

            # if it's new, append; if resume-updated, it's already in results by reference
            if not (args.resume and out.get("edit_image") in by_edit):
                results.append(out)
                by_edit[out["edit_image"]] = out

            updated += 1
            save_json(args.output, results)

    print(f"Done. Updated/added: {updated}. Output: {args.output}")


if __name__ == "__main__":
    main()
