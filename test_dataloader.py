from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from data_utils.dataset import create_dataloader

disable_torch_init()
model_name = get_model_name_from_path("/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="/home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b",
    model_name = model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device="cuda:0"
)

dataloader = create_dataloader(
    data_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data",
    annotation_path="/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation",
    data_config=model.config,
    image_processor=image_processor,
    tokenizer=tokenizer,
    batch_size=4,
    mode="train"
)

for batch in dataloader["train"]:
    # print(batch)
    print(batch['input_ids'])
    print(batch['input_ids'].shape)
    print(batch['image_tensor'].shape)
    print(batch['attention_masks'].shape)
    print(batch['mask_tensor'].shape)
    print(batch['answers_ids'].shape)
    print("====================")