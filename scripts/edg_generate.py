"""
scripts/edg_generate.py

EDG (Eq.1): offline enriched description generation. NOT part of the trainable
graph. Produces enriched.json {image_name: caption} consumed by RST2IDataset.

Backends (Table 5): InternVL2.5-8B (paper default), Qwen2.5-VL, GPT-4o-style API,
plus an 'echo' no-dependency fallback for pipeline bring-up.

  python scripts/edg_generate.py --images_dir /data/rsicd/images \
      --coarse_captions coarse.json --scene_prompts scenes.json \
      --out enriched.json --backend internvl
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

STRUCTURED_PROMPT = (
    "You are annotating a remote sensing aerial image of scene type '{scene}'. "
    "Coarse caption: '{coarse}'. Write one detailed factual description covering "
    "the primary scene, salient objects, their spatial layout and relationships, "
    "environmental context, and notable visual attributes. Use concise "
    "third-person prose. Do not invent objects not visible. Output description only."
)


class EchoBackend:
    def caption(self, path, coarse, scene):
        base = coarse or f"an aerial view of a {scene}"
        return (f"The image is an aerial view of a {scene}. {base}. Salient "
                f"objects are distributed across the scene with surrounding "
                f"context and varied land cover.")


class InternVLBackend:
    def __init__(self, model_path="OpenGVLab/InternVL2_5-8B"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False)

    def _img(self, path):
        import torchvision.transforms as T
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((448, 448))
        tf = T.Compose([T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
        return tf(im).unsqueeze(0).to(self.torch.bfloat16).cuda()

    def caption(self, path, coarse, scene):
        prompt = STRUCTURED_PROMPT.format(scene=scene, coarse=coarse or "(none)")
        pv = self._img(path)
        return self.model.chat(self.tok, pv, "<image>\n" + prompt,
                               dict(max_new_tokens=256, do_sample=False)).strip()


class QwenVLBackend:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto").eval()
        self.proc = AutoProcessor.from_pretrained(model_path)

    def caption(self, path, coarse, scene):
        from qwen_vl_utils import process_vision_info
        prompt = STRUCTURED_PROMPT.format(scene=scene, coarse=coarse or "(none)")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": path}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = self.proc(text=[text], images=imgs, videos=vids, padding=True,
                        return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=256)
        trim = out[:, inp.input_ids.shape[1]:]
        return self.proc.batch_decode(trim, skip_special_tokens=True)[0].strip()


class APIBackend:
    """OpenAI-compatible multimodal API backend (requires OPENAI_API_KEY)."""

    def __init__(self, model_path="gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model_path

    def caption(self, path, coarse, scene):
        prompt = STRUCTURED_PROMPT.format(scene=scene, coarse=coarse or "(none)")
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{encoded}"
                    }},
                ],
            }],
        )
        return response.choices[0].message.content.strip()


BACKENDS = {
    "echo": EchoBackend,
    "internvl": InternVLBackend,
    "qwen": QwenVLBackend,
    "api": APIBackend,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--coarse_captions", default=None)
    ap.add_argument("--scene_prompts", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="echo", choices=list(BACKENDS))
    ap.add_argument("--model", default=None, help="optional backend model path/id")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--failures", default=None)
    a = ap.parse_args()

    coarse = json.loads(Path(a.coarse_captions).read_text(encoding="utf-8")) if a.coarse_captions else {}
    scenes = json.loads(Path(a.scene_prompts).read_text(encoding="utf-8")) if a.scene_prompts else {}
    be = BACKENDS[a.backend](a.model) if a.model else BACKENDS[a.backend]()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {} if a.overwrite or not out_path.exists() else json.loads(
        out_path.read_text(encoding="utf-8")
    )
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    names = sorted(
        name for name in os.listdir(a.images_dir)
        if Path(name).suffix.lower() in extensions
    )
    failures = []
    for i, name in enumerate(names):
        if name in out and not a.overwrite:
            continue
        try:
            caption = be.caption(os.path.join(a.images_dir, name),
                                 coarse.get(name, ""), scenes.get(name, "scene"))
            if not caption or not caption.strip():
                raise ValueError("backend returned an empty description")
            out[name] = caption.strip()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failures.append({"image": name, "error": repr(exc)})
            print(f"failed {name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(names)}")
            out_path.write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        failure_path = Path(a.failures or f"{a.out}.failures.json")
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"{len(failures)} images failed; see {failure_path}")
    print(f"wrote {len(out)} -> {a.out}")


if __name__ == "__main__":
    main()
