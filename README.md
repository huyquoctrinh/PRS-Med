# PRS-Med 

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-orange.svg)]([huyquoctrinh/PRS-Med](https://huggingface.co/huyquoctrinh/PRS-Med))
[![arXiv](https://img.shields.io/badge/arXiv-2505.11872-b31b1b.svg)](https://arxiv.org/abs/2505.11872)

**PRS-Med** is a modular framework for training and inference of segmentation models powered by large language models (LLMs). It integrates components like LLaVA, Segment Anything, and TinySAM to perform multimodal segmentation tasks.

## Updated

- 23/09/2025: Published the PRS-Med dataset, including Medical Position QA, Multiple Choice QA about position and medical reasoning
- 01/06/2025: Updated repository of PRS-Med

## 🔧 Features

* Support Position Reasoning Segmentation task
* Support evaluation tool for reasoning
* Support evaluation tool for segmentation
* Support training and inference of the model

---

## Dataset

For the PRS-Med dataset, it is available in:

- Google Drive: [Part 1](https://drive.google.com/file/d/1vY6UD4bfccdIDRpwpG_nVZ9r1vSYPRd1/view?usp=drive_link), [Part 2](https://drive.google.com/file/d/1Lt0y9UiQFDQ9PgnW1oYy1hW6I211Glot/view?usp=drive_link), [Annotations](https://drive.google.com/drive/folders/1VyFqcfDbvrtYBA13ZkDz0scQmehLYYzt?usp=drive_link)
- [Hugging Face Dataset:](https://huggingface.co/datasets/huyquoctrinh/PRS-Med) [![Hugging Face](https://img.shields.io/badge/🤗-Dataset%20Repo-yellow.svg)](https://huggingface.co/datasets/huyquoctrinh/PRS-Med)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/huyquoctrinh/LLM_segmentation.git](https://github.com/huyquoctrinh/PRS-Med/tree/main)
cd PRS-Med
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights (Will be updated soon)

```bash
bash download.sh
```
---

## 🏋️‍♂️ Training (Full code is not completely updated yet)

Use `train.py` to train a segmentation model.

### Example:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ddp.py \
  --model_path /path/to/your/based_model \
  --data_path /path/to/your/images/data \
  --annotation_path /path/to/your/annotations \
  --batch_size 4 --epochs 50 --save_dir /path/to/your/save/dir \
  --grad_accum_steps 8 \
  --grad_clip_norm 1.0 \
```


## 🧪 Inference

Use `infer.py` to perform inference on images.

### Example:

```bash
python infer_full.py \
```

**Note:** Please updated your checkpoint inside that folder to match with what your trained model

---

## 📊 Evaluation (will be updated later)

Use the scripts in the `evaluation/` directory to assess model performance. There are two evaluation tools for reasoning and segmentation. 

## 📈 Visualization (will be updated later)

Use `visualize.py` to visualize segmentation results.

### Example:

```bash
python visualize.py \
  --image path/to/image.jpg \
  --mask path/to/mask.png
```



---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more details, visit the [LLM\_segmentation GitHub repository](https://github.com/huyquoctrinh/LLM_segmentation).

---

## Citation

```
@article{trinh2025prs,
  title   = {Prs-med: Position reasoning segmentation with vision-language model in medical imaging},
  author  = {Trinh, Quoc-Huy and Nguyen, Minh-Van and Zeng, Jung and Bagci, Ulas and Jha, Debesh},
  journal = {arXiv preprint arXiv:2505.11872},
  year    = {2025}
}

```

[1]: https://github.com/deep-diver/LLM-Serve?utm_source=chatgpt.com "GitHub - deep-diver/LLM-Serve: This repository provides a framework to ..."
[2]: https://github.com/yakhyo/crack-segmentation/blob/main/inference.py?utm_source=chatgpt.com "crack-segmentation/inference.py at main - GitHub"
