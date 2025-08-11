import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch.nn as nn
from tinysam import sam_model_registry, SamHierarchicalMaskGenerator
import torch 
from segment_model.mask_decoder_v5 import PromptedMaskDecoder
import peft
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
import math 
from sam_med.segment_anything import sam_model_registry
from argparse import Namespace
# from segment_anything import sam_model_registry


def custom_lora_init(module):
    if hasattr(module, "lora_A"):
        nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
    if hasattr(module, "lora_B"):
        nn.init.zeros_(module.lora_B.weight)

class SamMedImageEncoder(nn.Module):
    def __init__(self):
        super(SamMedImageEncoder, self).__init__()
        args = Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/home/mamba/ML_project/Testing/Huy/gaussian_splatting/SAM-Med2D/pretrained/sam-med2d_b.pth"
        self.sam = sam_model_registry["vit_b"](args)
        self.image_encoder = self.sam.image_encoder

    def forward(self, inputs):
        # with torch.no_grad():
        return self.image_encoder(inputs)

class ImageEncoder(nn.Module):
    def __init__(self, model_type, checkpoint_path):
        super(ImageEncoder, self).__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = self.sam.image_encoder

    def forward(self, inputs):
        # with torch.no_grad():
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
            lora_alpha=16,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            inference_mode=False,
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

        self.model = get_peft_model(self.base_model, lora_config)
        # self.model.to(dtype=torch.float32)
        if self.training:
            self.model.to(dtype=torch.bfloat16)

        self.mask_decoder = PromptedMaskDecoder()
        self.image_encoder = SamMedImageEncoder()
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        if self.training:
            self.mask_decoder.train()
            self.image_encoder.train()
        # self.cls = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(256, 6)
        # )
        torch.nn.init.xavier_uniform_(self.cls[2].weight)
        torch.nn.init.ones_(self.cls[2].bias)

    def get_model_utils(self):
        return self.tokenizer, self.image_processor, self.context_len, self.base_model.config
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path + "/lora_adapter")
        self.tokenizer.save_pretrained(save_path + "/lora_adapter")
        torch.save(self.image_encoder.state_dict(), save_path + "/image_encoder.pth")
        torch.save(self.mask_decoder.state_dict(), save_path + "/mask_decoder.pth")
        # torch.save(self.cls.state_dict(), save_path + "/cls.pth")

    def load_model(self, load_path):
        print("Loading model from:", load_path)
        self.tokenizer = self.tokenizer.from_pretrained(load_path + "/lora_adapter/")
        self.mask_decoder.load_state_dict(torch.load(load_path + "/mask_decoder.pth"))
        self.image_encoder.load_state_dict(torch.load(load_path + "/image_encoder.pth"))
        self.model = PeftModel.from_pretrained(self.model, load_path + "/lora_adapter/")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.mask_decoder.to(self.device)
        self.mask_decoder.eval()
        self.model = self.model.merge_and_unload()
        self.image_encoder.eval()
        self.model.eval()
        return self.tokenizer
    
    def generate(
        self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        input_ids_for_seg=None,
        attention_mask = None,
        temperature=0.0001,
        max_new_tokens=512,
        top_p=0.95
    ):
        self.image_encoder.eval()
        self.model.eval()
        self.mask_decoder.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs = input_ids,
                images = image_tensor_for_vlm,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )

            image_embedding = self.image_encoder(image_tensor_for_image_enc)
            prompt_embedding = self.base_model.extract_last_hidden_state(
                input_ids = input_ids_for_seg if input_ids_for_seg is not None else input_ids,
                images = image_tensor_for_vlm,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )["hidden_states"][-1]
            final_mask = self.mask_decoder(
                image_embedding, prompt_embedding
            )
        return final_mask, output_ids

    def forward(self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        answers=None
    ):
        if self.training:
            self.model.to(dtype=torch.bfloat16)
        else:
            self.model.to(dtype=torch.float16)

        # with torch.no_grad():
        output_dict = self.model(
            input_ids = input_ids,
            images = image_tensor_for_vlm,
            return_dict = True,
            output_hidden_states=True,
            return_loss = False
        )
        prompt_embedding = output_dict["hidden_states"][-1]
        logits = output_dict["logits"]

        image_embedding = self.image_encoder(image_tensor_for_image_enc)
        # print(image_embedding)
        output_cls = self.cls(image_embedding)
        # print("Output cls:", output_cls)
        final_mask = self.mask_decoder(
            image_embedding, prompt_embedding
        )
        if self.training:
            logit_loss = self.loss(logits.view(-1, logits.size(-1)), answers.view(-1))
            return final_mask, output_cls, logit_loss
        else:
            return final_mask, logits

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