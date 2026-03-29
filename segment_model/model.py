from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from transformers import AutoTokenizer
import torch.nn as nn
from tinysam import sam_model_registry
import torch
from segment_model.mask_decoder import PromptedMaskDecoder
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


class ImageEncoder(nn.Module):
    def __init__(self, model_type, checkpoint_path):
        super(ImageEncoder, self).__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = self.sam.image_encoder

    def forward(self, inputs):
        return self.image_encoder(inputs)


class LLMSeg(nn.Module):
    def __init__(
            self,
            model_path,
            model_base=None,
            load_8bit=False,
            load_4bit=False,
            device="cuda:0"
        ):
        super(LLMSeg, self).__init__()
        disable_torch_init()
        self.device = device

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.base_model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=self.device
        )
        self.base_model.eval()

        if self.training:
            self.base_model.to(torch.bfloat16)
        else:
            self.base_model.to(torch.float16)

        self.model = get_peft_model(self.base_model, lora_config)
        self.mask_decoder = PromptedMaskDecoder()
        self.image_encoder = ImageEncoder(
            model_type="vit_t",
            checkpoint_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/sam_ckpts/tinysam_42.3.pth"
        )

        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 6)
        )

        if self.training:
            self.model.to(dtype=torch.bfloat16)
            self.image_encoder.to(dtype=torch.bfloat16)
            self.mask_decoder.to(dtype=torch.bfloat16)
            self.cls.to(dtype=torch.bfloat16)
            torch.nn.init.xavier_uniform_(self.cls[2].weight)
            torch.nn.init.ones_(self.cls[2].bias)
        else:
            self.model.to(dtype=torch.float16)
            self.image_encoder.to(dtype=torch.float16)
            self.mask_decoder.to(dtype=torch.float16)
            self.cls.to(dtype=torch.float16)

    def get_model_utils(self):
        return self.tokenizer, self.image_processor, self.context_len, self.base_model.config

    def save_model(self, save_path):
        self.model.save_pretrained(save_path + "/lora_adapter")
        self.tokenizer.save_pretrained(save_path + "/tokenizer")
        torch.save(self.image_encoder.state_dict(), save_path + "/image_encoder.pth")
        torch.save(self.mask_decoder.state_dict(), save_path + "/mask_decoder.pth")
        torch.save(self.cls.state_dict(), save_path + "/cls.pth")

    def load_model(self, load_path):
        print("Loading model from:", load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path + "/tokenizer/")
        self.mask_decoder.load_state_dict(
            torch.load(load_path + "/mask_decoder.pth", map_location=self.device))
        self.image_encoder.load_state_dict(
            torch.load(load_path + "/image_encoder.pth", map_location=self.device))
        self.cls.load_state_dict(
            torch.load(load_path + "/cls.pth", map_location=self.device))
        self.model = PeftModel.from_pretrained(self.base_model, load_path + "/lora_adapter/")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.merge_and_unload()
        self.base_model = self.model
        # Move all components to device and set to eval/float16
        self.model.to(self.device, dtype=torch.float16).eval()
        self.image_encoder.to(self.device, dtype=torch.float16).eval()
        self.mask_decoder.to(self.device, dtype=torch.float16).eval()
        self.cls.to(self.device, dtype=torch.float16).eval()
        return self.tokenizer

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        input_ids_for_seg=None,
        attention_mask=None,
        temperature=0.1,
        max_new_tokens=512,
        top_p=0.95
    ):
        self.image_encoder.eval()
        self.model.eval()
        self.mask_decoder.eval()

        output = self.model.generate(
            inputs=input_ids,
            images=image_tensor_for_vlm,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        prompt_embedding = self.model.extract_last_hidden_state(
            input_ids=input_ids_for_seg if input_ids_for_seg is not None else input_ids,
            images=image_tensor_for_vlm,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p
        )["hidden_states"][-1]

        output_ids = output["sequences"]
        enc_dtype = next(self.image_encoder.parameters()).dtype
        image_embedding = self.image_encoder(
            image_tensor_for_image_enc.to(enc_dtype))
        # mask_decoder internally casts to float32, so ensure
        # weights match by running decoder in float32
        self.mask_decoder.float()
        final_mask = self.mask_decoder(
            image_embedding.float(), prompt_embedding.float())
        return final_mask, output_ids

    def forward(
        self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        attention_mask=None,
        answers=None,
        temperature=0.0001,
        max_new_tokens=512,
        top_p=0.95
    ):
        if self.training:
            self.model.to(dtype=torch.bfloat16)
        else:
            self.model.to(dtype=torch.float16)

        output = self.model(
            input_ids=answers,
            attention_mask=attention_mask,
            images=image_tensor_for_vlm,
            use_cache=False,
            labels=answers,
            return_dict=True,
            output_hidden_states=True,
        )

        prompt_embedding = output["hidden_states"][-1]
        logit_loss = output["loss"]
        enc_dtype = next(self.image_encoder.parameters()).dtype
        image_embedding = self.image_encoder(
            image_tensor_for_image_enc.to(enc_dtype))
        output_cls = self.cls(image_embedding)
        final_mask = self.mask_decoder(image_embedding, prompt_embedding)
        return final_mask, output_cls, logit_loss


def build_llm_seg(
        model_path,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda:0"
):
    llm_seg = LLMSeg(
        model_path=model_path,
        model_base=model_base,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device
    )

    tokenizer, image_processor, context_len, config = llm_seg.get_model_utils()
    return llm_seg, tokenizer, image_processor, config
