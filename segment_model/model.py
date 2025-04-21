import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch.nn as nn
from tinysam import sam_model_registry, SamHierarchicalMaskGenerator
import torch 
from segment_model.mask_decoder import PromptedMaskDecoder

class ImageEncoder(nn.Module):
    def __init__(self, model_type, checkpoint_path):
        super(ImageEncoder, self).__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.eval()
        self.image_encoder = self.sam.image_encoder

    def forward(self, inputs):
        with torch.no_grad():
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

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=self.device
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # self.model = self.model.to_fp32()

        self.mask_decoder = PromptedMaskDecoder()

        self.image_encoder = ImageEncoder(
            model_type="vit_t",
            checkpoint_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/sam_ckpts/tinysam_42.3.pth"
        ).to(self.device)
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def get_model_utils(self):
        return self.tokenizer, self.image_processor, self.context_len, self.model.config

    def forward(self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        temperature=0.0001,
        max_new_tokens=512,
        top_p=0.95
    ):
        with torch.no_grad():
            prompt_embedding = self.model.extract_last_hidden_state(
                input_ids,
                images=image_tensor_for_vlm,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )["hidden_states"][-1]
        # print("======================")
        # print("Input ids:", input_ids)
        # print("Global embedding:", torch.mean(prompt_embedding, dim=1))
        # print("======================")
        with torch.no_grad():
            image_embedding = self.image_encoder(image_tensor_for_image_enc)
        # print(image_embedding)
        final_mask = self.mask_decoder(
            image_embedding, prompt_embedding
        )
        return final_mask

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