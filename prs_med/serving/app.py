"""Gradio server for the conversational [SEG] segmentation model.

Upload a medical image and chat about it. The model answers in text, and when
you ask it to segment, it emits a [SEG] token internally and returns a mask
overlay. Descriptive questions return text only -- segmentation is gated on the
request, not produced every turn.

Usage:
    python -m serving.app --checkpoint training_results/train_chat_seg/llm_seg_chat_10
    python -m serving.app --share          # public link
"""
import argparse
import os
import sys
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# --- gradio_client 1.5.1 workaround (must run before `import gradio`) --------
# JSON Schema allows a bare `true`/`false` as a schema (e.g.
# `additionalProperties: true`). gradio_client's walker assumes a dict and does
# `"const" in schema`, which raises "argument of type 'bool' is not iterable".
# That exception fires inside the request handler, so gradio's localhost
# self-check at launch fails and the server exits. Patch it here rather than
# upgrading gradio in this shared environment.
import gradio_client.utils as _gcu  # noqa: E402

_orig_json_schema = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type


def _safe_get_type(schema):
    if isinstance(schema, bool) or not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


def _safe_json_schema(schema, defs=None):
    if isinstance(schema, bool) or not isinstance(schema, dict):
        return "Any"
    try:
        return _orig_json_schema(schema, defs)
    except TypeError:
        return "Any"


_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_json_schema
# ----------------------------------------------------------------------------

import gradio as gr  # noqa: E402

from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from prs_med.models.llm_seg_chat import build_llm_seg_chat  # noqa: E402

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None
CONFIG = None
DEVICE = "cuda:0"
# Inference is not re-entrant (single model instance, single GPU) -- serialize.
LOCK = threading.Lock()

SAM_TF = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EXAMPLES = [
    "What organs are visible in this slice?",
    "Segment the liver and the spleen.",
    "Now segment just the spleen.",
    "Can you figure out both kidneys for me?",
    "Point out the pancreas and the stomach.",
    "Segment the tumor for me.",
]


def load_model(model_path, checkpoint, device):
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, CONFIG, DEVICE
    DEVICE = device
    print(f"Loading base model from {model_path} ...", flush=True)
    MODEL, TOKENIZER, IMAGE_PROCESSOR, CONFIG = build_llm_seg_chat(
        model_path=model_path, device=device, is_training=False)
    if checkpoint:
        print(f"Loading checkpoint {checkpoint} ...", flush=True)
        TOKENIZER = MODEL.load_checkpoint(checkpoint)
    else:
        print("[warn] no --checkpoint given; [SEG] is untrained", flush=True)
    MODEL.eval()
    print(f"Ready. [SEG] id = {MODEL.seg_token_idx}", flush=True)


def encode_image(pil):
    """CLIP tensor for the VLM + SAM tensor for the mask encoder."""
    pil = pil.convert("RGB")
    vlm = process_images([pil], IMAGE_PROCESSOR, CONFIG).to(DEVICE, torch.float16)
    sam = SAM_TF(pil).unsqueeze(0).to(DEVICE, torch.float16)
    return vlm, sam, pil


def build_prompt(history, user_msg):
    """mistral_instruct rendering of the whole dialogue so far."""
    parts = []
    for i, (q, a) in enumerate(history):
        img = DEFAULT_IMAGE_TOKEN + "\n" if i == 0 else ""
        parts.append(f"[INST] {img}{q} [/INST] {a}</s>")
    img = DEFAULT_IMAGE_TOKEN + "\n" if not history else ""
    parts.append(f"[INST] {img}{user_msg} [/INST]")
    return "".join(parts)


# Distinct overlay colours (BGR-agnostic RGB) cycled per mask.
_PALETTE = [
    (255, 60, 60), (60, 200, 60), (70, 120, 255), (255, 210, 40),
    (220, 80, 255), (0, 220, 220), (255, 140, 0), (150, 255, 90),
]

# Forcing a token for an organ that isn't on the slice yields a near-empty mask.
# Drop masks below this foreground fraction (they are "organ absent"), but always
# keep at least the largest so an explicit single-organ ask still shows something.
_EMPTY_MASK_FRAC = 0.006


def _drop_empty_masks(masks, token_ids, threshold):
    """Filter out near-empty (absent-organ) masks; keep >=1."""
    if masks is None:
        return None, token_ids
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)
    fr = [(masks[i].sigmoid() > threshold).float().mean().item()
          for i in range(masks.shape[0])]
    keep = [i for i, f in enumerate(fr) if f >= _EMPTY_MASK_FRAC]
    if not keep:
        keep = [int(max(range(len(fr)), key=lambda i: fr[i]))]
    if len(keep) == masks.shape[0]:
        return masks, token_ids
    idx = torch.tensor(keep, device=masks.device)
    tk = [token_ids[i] for i in keep] if token_ids else token_ids
    return masks.index_select(0, idx), tk


def overlay_masks(pil, masks, token_ids, threshold=0.5, alpha=0.45):
    """Blend one or more predicted masks over the image, each a distinct colour,
    with a legend (colour -> organ). Returns (overlay, heatmap, [(name, frac)])."""
    rgb = np.array(pil.convert("RGB"))
    out = rgb.copy().astype(np.float32)
    heat_acc = np.zeros(pil.size[::-1], np.float32)
    legend = []
    if masks is None:
        return Image.fromarray(out.astype(np.uint8)), None, legend
    if masks.dim() == 3:                       # (1,H,W) -> (1,1,H,W)
        masks = masks.unsqueeze(0)
    n = masks.shape[0]
    id2name = {v: k for k, v in getattr(MODEL, "seg_token_ids", {}).items()}
    for i in range(n):
        prob = masks[i].sigmoid().float().cpu().numpy().squeeze()
        prob = cv2.resize(prob, pil.size, interpolation=cv2.INTER_LINEAR)
        heat_acc = np.maximum(heat_acc, np.clip(prob, 0, 1))
        binary = prob > threshold
        colour = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        if binary.any():
            out[binary] = (1 - alpha) * out[binary] + alpha * colour
            contours, _ = cv2.findContours(
                binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, tuple(int(c) for c in colour), 2)
        tok = token_ids[i] if token_ids and i < len(token_ids) else None
        name = _SEG.organ_name(id2name.get(tok, "")) if tok is not None else f"mask {i}"
        legend.append((name or f"mask {i}", float(binary.mean()),
                       tuple(int(c) for c in colour)))
    heat = (heat_acc * 255).astype(np.uint8)
    return Image.fromarray(out.astype(np.uint8)), Image.fromarray(heat), legend


import re as _re
import prs_med.models.seg_vocab as _SEG

_SEG_INTENT = _re.compile(
    r"\b(segment|segmentation|outline|delineate|mask|highlight|trace|contour|"
    r"locate|find|identify|point out|figure out|map out|mark|"
    r"give (me )?the (position|location)|position of|where (exactly )?(is|are)|"
    r"show me (the|its|where))\b", _re.I)

# Non-lobe organ / lesion keyword -> token. All patterns are order-independent
# (lung lobes are handled separately by _lobe_tokens so word order like
# "left lung upper lobe" vs "left upper lobe" both resolve).
_KW2TOK = [
    (r"left kidney|kidney[ _]?left", "[SEG-kidney_left]"),
    (r"right kidney|kidney[ _]?right", "[SEG-kidney_right]"),
    (r"brain tumou?r|glioma|tumou?r in the brain", "[SEG-brain_tumor]"),
    (r"breast tumou?r|breast (mass|lesion)", "[SEG-breast_tumor]"),
    (r"\bpolyp", "[SEG-polyp]"),
    (r"skin lesion|melanoma|nevus|mole", "[SEG-skin_lesion]"),
    (r"\bliver\b", "[SEG-liver]"), (r"\bspleen\b", "[SEG-spleen]"),
    (r"\bpancreas\b", "[SEG-pancreas]"),
    (r"\bstomach\b", "[SEG-stomach]"), (r"gall ?bladder", "[SEG-gallbladder]"),
    (r"urinary bladder|\bbladder\b", "[SEG-urinary_bladder]"),
    (r"\bheart\b|cardiac", "[SEG-heart]"), (r"\bcolon\b", "[SEG-colon]"),
    (r"small bowel|small intestine", "[SEG-small_bowel]"),
    (r"\bduodenum\b", "[SEG-duodenum]"), (r"\besophagus\b|oesophagus", "[SEG-esophagus]"),
    (r"\btrachea\b", "[SEG-trachea]"), (r"thyroid", "[SEG-thyroid_gland]"),
    (r"\bprostate\b", "[SEG-prostate]"),
    (r"\bbrain\b(?!\s+tumou?r)", "[SEG-brain]"),   # not "brain tumor"
]

# Anaphoric segmentation asks ("segment it / them / all organs") carry no organ
# name, so the target must be resolved from the conversation instead.
_ANAPHORIC = _re.compile(
    r"\b(it|them|those|these|that|this|both|again|"
    r"all( of them| the organs| organs| structures)?|everything|"
    r"the (organs?|structures?|regions?|targets?|areas?))\b", _re.I)

_LOBE_TOKEN = {
    ("left", "upper"): "[SEG-lung_upper_lobe_left]",
    ("right", "upper"): "[SEG-lung_upper_lobe_right]",
    ("left", "lower"): "[SEG-lung_lower_lobe_left]",
    ("right", "lower"): "[SEG-lung_lower_lobe_right]",
    ("right", "middle"): "[SEG-lung_middle_lobe_right]",  # no left middle lobe
}
# side + level ADJACENT to "lobe" (any of the phrasings), so two lobes in one
# clause don't cross-produce a spurious third. "lung" may sit between the words.
# "lung" may sit between side+level OR between level+lobe:
# "right lung lower lobe", "right lower lung lobe", "right lower lobe" all match.
_LOBE_RE = _re.compile(
    r"\b(left|right)\s+(?:lung\s+)?(upper|lower|middle)\s+(?:lung\s+)?lobe", _re.I)
_LOBE_RE_OF = _re.compile(
    r"\b(upper|lower|middle)\s+(?:lung\s+)?lobe\s+of\s+(?:the\s+)?(left|right)", _re.I)
_MIDDLE_RE = _re.compile(r"\bmiddle\s+(?:lung\s+)?lobe", _re.I)


def _lobe_tokens(text):
    """Lung-lobe detection by side+level ADJACENT to 'lobe' (word-order tolerant:
    'left lung upper lobe', 'left upper lobe', 'upper lobe of the left lung')."""
    text = text or ""
    out = []

    def add(side, level):
        t = ("[SEG-lung_middle_lobe_right]" if level == "middle"
             else _LOBE_TOKEN.get((side, level)))
        if t and t not in out:
            out.append(t)

    for m in _LOBE_RE.finditer(text):
        add(m.group(1).lower(), m.group(2).lower())
    for m in _LOBE_RE_OF.finditer(text):
        add(m.group(2).lower(), m.group(1).lower())
    if _MIDDLE_RE.search(text) and "[SEG-lung_middle_lobe_right]" not in out:
        out.append("[SEG-lung_middle_lobe_right]")
    return out


def _organ_tokens(text):
    """All per-type tokens named in `text`, ordered & de-duped."""
    text = text or ""
    low = text.lower()
    picked = []

    def add(t):
        if t not in picked:
            picked.append(t)

    for t in _lobe_tokens(text):
        add(t)
    for pat, tok in _KW2TOK:
        if _re.search(pat, text, _re.I):
            add(tok)
    # plural / side-less kidney -> both kidneys
    if _re.search(r"\bkidneys\b", low) or (
        _re.search(r"\bkidney\b", low)
        and not any("kidney" in t for t in picked)
    ):
        add("[SEG-kidney_left]")
        add("[SEG-kidney_right]")
    # whole lung field only when no specific lobe was requested
    if _re.search(r"lung field|\blungs?\b", low) and "lobe" not in low:
        add("[SEG-lung_fields]")
    return picked


_REDO_RE = _re.compile(
    r"\b(redo|regenerate|retry|again|re-?segment|re-?do|one more time|try again|"
    r"do it again|repeat)\b", _re.I)

_EXPAND_RE = _re.compile(
    r"\b(bigger|larger|more|expand|wider|include more|grow|enlarge|"
    r"too small|too tight|not enough)\b", _re.I)

_SHRINK_RE = _re.compile(
    r"\b(smaller|less|tighter|shrink|reduce|narrow|too big|too large|"
    r"too much|overl(y|apping))\b", _re.I)


def _wants_segmentation(msg):
    return bool(_SEG_INTENT.search(msg or ""))


def _resolve_seg_tokens(user_msg, reply, history, last_seg_tokens=None):
    """Full set of tokens to segment. Prefer organs named in the user's message;
    if the ask is anaphoric ('segment it / all organs'), resolve from the model's
    reply plus prior assistant turns (which name the visible organs).
    On redo requests with no new organ, reuse last_seg_tokens."""
    direct = _organ_tokens(user_msg)
    if direct:
        return direct
    if last_seg_tokens and _REDO_RE.search(user_msg or ""):
        return list(last_seg_tokens)
    ctx = (reply or "") + " " + " ".join(a for _, a in (history or []))
    toks = _organ_tokens(ctx)
    return toks or ["[SEG-lung_fields]"]


def _adjust_threshold(user_msg, base_threshold):
    """Shift threshold for expand/shrink refinement requests."""
    if _EXPAND_RE.search(user_msg or ""):
        return max(0.1, base_threshold - 0.15)
    if _SHRINK_RE.search(user_msg or ""):
        return min(0.9, base_threshold + 0.15)
    return base_threshold


def respond(user_msg, chat, history, image, cache, temperature, max_new_tokens,
            threshold, last_seg_tokens):
    if image is None:
        chat = chat + [{"role": "assistant", "content": "Please upload an image first."}]
        return chat, history, cache, None, None, "", last_seg_tokens
    if not (user_msg or "").strip():
        return chat, history, cache, None, None, "", last_seg_tokens

    with LOCK:
        if cache is None:
            cache = encode_image(image)
        vlm, sam, pil = cache

        prompt = build_prompt(history, user_msg)
        input_ids = tokenizer_image_token(
            prompt, TOKENIZER, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(DEVICE)

        is_redo = bool(_REDO_RE.search(user_msg or ""))
        gen_temp = float(temperature)
        if is_redo and gen_temp < 0.3:
            gen_temp = 0.3

        _, _, output_ids = MODEL.generate(
            input_ids=input_ids,
            image_tensor_for_vlm=vlm,
            image_tensor_for_image_enc=sam,
            temperature=gen_temp,
            max_new_tokens=int(max_new_tokens),
        )

        raw = TOKENIZER.decode(output_ids[0], skip_special_tokens=False)
        for tok in (TOKENIZER.eos_token, TOKENIZER.bos_token, "<unk>", "<s>", "</s>"):
            if tok:
                raw = raw.replace(tok, "")
        reply = raw.strip()
        for st in _SEG.SEG_TOKENS + ["[SEG_TUMOR]", "[SEG_ORGAN]"]:
            reply = reply.replace(st, "")
        shown = reply.strip()

        masks, token_ids = None, None
        if _wants_segmentation(user_msg):
            seg_tokens = _resolve_seg_tokens(
                user_msg, shown, history, last_seg_tokens)
            last_seg_tokens = list(seg_tokens)
            full = build_prompt(history, user_msg) + " " + shown
            ids = tokenizer_image_token(
                full, TOKENIZER, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                masks, token_ids = MODEL.segment_forced(ids, vlm, sam, seg_tokens)
            effective_thresh = _adjust_threshold(user_msg, float(threshold))
            masks, token_ids = _drop_empty_masks(masks, token_ids, effective_thresh)

    if masks is not None:
        effective_thresh = _adjust_threshold(user_msg, float(threshold))
        ov, heat, legend = overlay_masks(pil, masks, token_ids,
                                         threshold=effective_thresh)
        parts = ", ".join(f"{name} {frac*100:.1f}%" for name, frac, _ in legend)
        status = f"Segmented: {parts}"
    else:
        ov, heat = None, None
        status = "No segmentation (text-only turn — ask it to segment to get a mask)"

    history = history + [(user_msg, reply)]
    chat = chat + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": shown or "(empty response)"},
    ]
    return chat, history, cache, ov, heat, status, last_seg_tokens


def on_new_image(image):
    """A new image invalidates the conversation and the cached encodings."""
    return [], [], None, None, None, "New image loaded — conversation reset.", None


def reset(image):
    return [], [], (None if image is None else None), None, None, "Conversation cleared.", None


def build_ui():
    with gr.Blocks(title="LLM-Seg Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# LLM-Seg — conversational medical segmentation\n"
            "Upload an image and chat. Ask a descriptive question and you get "
            "text only; ask it to **segment** and it returns a mask."
        )

        history = gr.State([])   # [(user, assistant_raw), ...]
        cache = gr.State(None)   # cached (vlm, sam, pil) for the current image
        last_seg_tokens = gr.State(None)  # seg tokens from the last segmentation turn

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Medical image", height=320)
                with gr.Accordion("Generation settings", open=False):
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                            label="Temperature (0 = greedy)")
                    max_new_tokens = gr.Slider(32, 512, value=256, step=32,
                                               label="Max new tokens")
                    threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                          label="Mask threshold")
                status = gr.Markdown("")

            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Conversation", height=420, type="messages")
                with gr.Row():
                    box = gr.Textbox(placeholder="Ask about the image, or say "
                                                 "'segment the tumor'...",
                                     show_label=False, scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(examples=EXAMPLES, inputs=box, label="Try")
                clear = gr.Button("Clear conversation")

            with gr.Column(scale=1):
                overlay = gr.Image(label="Mask overlay", height=320)
                heatmap = gr.Image(label="Probability map", height=220)

        outs = [chat, history, cache, overlay, heatmap, status, last_seg_tokens]
        ins = [box, chat, history, image, cache, temperature, max_new_tokens,
               threshold, last_seg_tokens]

        send.click(respond, ins, outs).then(lambda: "", None, box)
        box.submit(respond, ins, outs).then(lambda: "", None, box)
        image.change(on_new_image, [image], outs)
        clear.click(reset, [image], outs)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="training_results/train_chat_v6/llm_seg_chat_10")
    ap.add_argument("--model_path", default="weight/llava-med-v1.5-mistral-7b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Create a public gradio.live link.")
    args = ap.parse_args()

    load_model(args.model_path, args.checkpoint, args.device)
    demo = build_ui().queue(max_size=16)

    if args.share:
        # Only path that uses gradio's own launcher (needed for the tunnel).
        demo.launch(server_name=args.host, server_port=args.port, share=True,
                    show_api=False, show_error=True)
        return

    # Serve the ASGI app directly instead of demo.launch(). launch() performs a
    # url_ok() self-request and aborts with "localhost is not accessible" if it
    # fails -- which it does in this environment even though loopback works
    # fine. Mounting on uvicorn ourselves skips that check.
    import uvicorn
    from fastapi import FastAPI

    app = gr.mount_gradio_app(FastAPI(), demo, path="/")
    print(f"Serving on http://{args.host}:{args.port}  "
          f"(local: http://127.0.0.1:{args.port})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
