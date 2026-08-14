"""Conversational LLMSeg v1.5: multi-turn dialogue with [SEG]-gated segmentation.

Differences from v1 (LLMSeg):
  - adds 27 per-type [SEG] tokens; mask decoder driven by hidden state at that
    token instead of the whole sequence
  - takes separate input_ids/labels so LM loss covers assistant turns only
  - single-pass generate() instead of two-pass
  - MedSAM image encoder instead of TinySAM
  - SamConditionedMaskDecoder (fp32-safe) instead of PromptedMaskDecoder
"""
import os

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from prs_med.models.sam_conditioned_decoder import SamConditionedMaskDecoder
from prs_med.models.sam_encoder import MedSamImageEncoder
from prs_med.models.seg_vocab import SEG_TOKENS, SEG_TOKEN

DEFAULT_SAM_MODEL = "wanglab/medsam-vit-base"


class LLMSegChat(nn.Module):
    def __init__(
        self,
        model_path,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda:0",
        is_training=True,
        sam_model=DEFAULT_SAM_MODEL,
        freeze_image_encoder=False,
    ):
        super().__init__()
        disable_torch_init()
        self.device = device
        self.is_training = is_training
        dtype = torch.bfloat16 if is_training else torch.float16

        model_name = get_model_name_from_path(model_path)
        (self.tokenizer, self.base_model, self.image_processor,
         self.context_len) = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit, device=device
        )

        added = self.tokenizer.add_tokens(SEG_TOKENS, special_tokens=True)
        if added:
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.seg_token_ids = {t: self.tokenizer.convert_tokens_to_ids(t)
                              for t in SEG_TOKENS}
        self.seg_token_idx = self.seg_token_ids[SEG_TOKENS[0]]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        self.base_model.to(dtype)
        self.base_model.eval()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
            inference_mode=not is_training,
        )
        self.model = get_peft_model(self.base_model, lora_config)

        self.mask_decoder = SamConditionedMaskDecoder(sam_model)
        self.image_encoder = MedSamImageEncoder(sam_model)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 7)
        )
        if is_training:
            nn.init.xavier_uniform_(self.cls[2].weight)
            nn.init.ones_(self.cls[2].bias)

        for m in (self.model, self.image_encoder, self.cls):
            m.to(device=device, dtype=dtype)
        self.mask_decoder.to(device=device, dtype=torch.float32)

        self.image_encoder_frozen = freeze_image_encoder
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)
            self.image_encoder.eval()

    def get_model_utils(self):
        return (self.tokenizer, self.image_processor, self.context_len,
                self.base_model.config)

    def forward(
        self,
        input_ids,
        labels,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        attention_mask=None,
    ):
        """Returns (mask, seg_batch_idx, cls_logits, lm_loss)."""
        outputs, new_labels = self.model.forward_with_aligned_labels(
            input_ids=input_ids,
            labels=labels,
            images=image_tensor_for_vlm,
            attention_mask=attention_mask,
        )
        lm_loss = outputs.loss

        if self.image_encoder_frozen:
            with torch.no_grad():
                image_embedding = self.image_encoder(image_tensor_for_image_enc)
        else:
            image_embedding = self.image_encoder(image_tensor_for_image_enc)
        cls_logits = self.cls(image_embedding)

        seg_mask = torch.zeros_like(new_labels, dtype=torch.bool)
        for tid in self.seg_token_ids.values():
            seg_mask |= (new_labels == tid)
        if not bool(seg_mask.any()):
            return None, None, cls_logits, lm_loss

        hidden = outputs.hidden_states[-1]
        seg_embeds = hidden[seg_mask].unsqueeze(1)
        seg_batch_idx = seg_mask.nonzero(as_tuple=True)[0]
        self.last_seg_kind = new_labels[seg_mask].detach()

        mask = self.mask_decoder(image_embedding[seg_batch_idx], seg_embeds)
        return mask, seg_batch_idx, cls_logits, lm_loss

    @torch.no_grad()
    def segment_forced(self, input_ids, image_tensor_for_vlm,
                       image_tensor_for_image_enc, seg_token):
        """Produce masks by forcing [SEG] token(s) at the end of the input."""
        multi = not isinstance(seg_token, str)
        tokens = list(seg_token) if multi else [seg_token]
        seg_ids = [self.seg_token_ids.get(t, self.seg_token_idx) for t in tokens]
        seg_row = torch.tensor([seg_ids], dtype=input_ids.dtype,
                               device=input_ids.device).expand(input_ids.shape[0], -1)
        ids = torch.cat([input_ids, seg_row], dim=1)
        outputs, new_labels = self.model.forward_with_aligned_labels(
            input_ids=ids, labels=ids, images=image_tensor_for_vlm,
            attention_mask=None)
        seg_mask = torch.zeros_like(new_labels, dtype=torch.bool)
        for sid in set(seg_ids):
            seg_mask |= (new_labels == sid)
        if not bool(seg_mask.any()):
            return (None, []) if multi else None
        seg_embeds = outputs.hidden_states[-1][seg_mask].unsqueeze(1)
        fired_ids = new_labels[seg_mask].tolist()
        image_embedding = self.image_encoder(image_tensor_for_image_enc)
        img = image_embedding.expand(seg_embeds.shape[0], -1, -1, -1)
        masks = self.mask_decoder(img, seg_embeds)
        return (masks, fired_ids) if multi else masks

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        attention_mask=None,
        temperature=0.2,
        max_new_tokens=512,
        top_p=0.95,
    ):
        """Single-pass: generate reply and, if it contains [SEG], the mask(s).

        Returns (masks_or_None, token_ids_or_None, sequences).
        """
        self.model.eval()
        self.image_encoder.eval()
        self.mask_decoder.eval()

        outputs = self.model.generate(
            inputs=input_ids,
            images=image_tensor_for_vlm,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        sequences = outputs.sequences

        gen_ids = sequences[0]
        is_seg = torch.zeros_like(gen_ids, dtype=torch.bool)
        for tid in self.seg_token_ids.values():
            is_seg |= (gen_ids == tid)
        if not bool(is_seg.any()):
            return None, None, sequences

        full_ids = torch.cat([input_ids, sequences], dim=1)
        fwd, new_labels = self.model.forward_with_aligned_labels(
            input_ids=full_ids, labels=full_ids,
            images=image_tensor_for_vlm, attention_mask=None)
        seg_mask = torch.zeros_like(new_labels, dtype=torch.bool)
        for tid in self.seg_token_ids.values():
            seg_mask |= (new_labels == tid)
        if not bool(seg_mask.any()):
            return None, None, sequences
        seg_embed = fwd.hidden_states[-1][seg_mask].unsqueeze(1)
        token_ids = new_labels[seg_mask].tolist()

        image_embedding = self.image_encoder(image_tensor_for_image_enc)
        img = image_embedding.expand(seg_embed.shape[0], -1, -1, -1)
        masks = self.mask_decoder(img, seg_embed)
        return masks, token_ids, sequences

    def save_checkpoint(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path + "/lora_adapter")
        self.tokenizer.save_pretrained(save_path + "/tokenizer")
        torch.save(self.image_encoder.state_dict(), save_path + "/image_encoder.pth")
        torch.save(self.mask_decoder.state_dict(), save_path + "/mask_decoder.pth")
        torch.save(self.cls.state_dict(), save_path + "/cls.pth")

    def load_checkpoint(self, load_path):
        print("Loading model from:", load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path + "/tokenizer/")
        self.seg_token_ids = {t: self.tokenizer.convert_tokens_to_ids(t)
                              for t in SEG_TOKENS}
        self.seg_token_idx = self.seg_token_ids[SEG_TOKENS[0]]

        self.mask_decoder.load_state_dict(
            torch.load(load_path + "/mask_decoder.pth", map_location="cpu"))
        self.image_encoder.load_state_dict(
            torch.load(load_path + "/image_encoder.pth", map_location="cpu"))
        cls_path = load_path + "/cls.pth"
        if os.path.isfile(cls_path):
            try:
                self.cls.load_state_dict(torch.load(cls_path, map_location="cpu"))
            except RuntimeError:
                pass

        emb_rows = self.base_model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) != emb_rows:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        self.model = PeftModel.from_pretrained(
            self.base_model, load_path + "/lora_adapter/")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.merge_and_unload()
        self.base_model = self.model

        for m in (self.model, self.image_encoder, self.cls):
            m.to(device=self.device, dtype=torch.float16)
            m.eval()
        self.mask_decoder.to(device=self.device, dtype=torch.float32).eval()
        return self.tokenizer


def build_llm_seg_chat(
    model_path,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device="cuda:0",
    is_training=True,
    freeze_image_encoder=False,
):
    model = LLMSegChat(
        model_path=model_path,
        model_base=model_base,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device,
        is_training=is_training,
        freeze_image_encoder=freeze_image_encoder,
    )
    tokenizer, image_processor, context_len, config = model.get_model_utils()
    return model, tokenizer, image_processor, config
