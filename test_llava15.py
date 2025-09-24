import torch
from PIL import Image
from llava_ablation.model.builder import load_pretrained_model
from llava_ablation.conversation import conv_templates
from llava_ablation.mm_utils import process_images, tokenizer_image_token
from segment_model.model_ablation15 import build_llm_seg

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# torch.cuda.set_device(1)
# model_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava15"
model_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/model_16"
# Paths
llm_seg, tokenizer, image_processor, config = build_llm_seg(
        model_path, 
        model_base=None, 
        load_8bit=False, 
        load_4bit=False, 
        device="cuda:0"
)
print("trainable params:", count_parameters(llm_seg))
# model_name = "llava-v1.5-7b"
# model_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava15"  # local model folder
# image_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/visualized_image.png"
# question = "<image> What is inside the image?"

# # Load model
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path,
#     None,
#     model_name,
#     False,
#     False,
#     device="cuda:1"
# )

# print(model)
# print("Number of trainable parameters:", count_parameters(model))
# # # Prepare conversation template
# conv = conv_templates["llava_v1"].copy()
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()

# # Load and process image
# image = load_image_from_path(image_path)
# image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

# # Tokenize prompt
# input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX=32000, return_tensors='pt').unsqueeze(0).to(model.device)

# # Run inference
# with torch.inference_mode():
#     output_ids = model.generate(
#         input_ids,
#         images=image_tensor,
#         do_sample=True,
#         temperature=0.2,
#         max_new_tokens=256
#     )

# # Decode output
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
# print("Assistant:", response)
