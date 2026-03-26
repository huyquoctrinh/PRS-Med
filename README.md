<h1 align="center">PRS-Med: Position Reasoning Segmentation in Medical Imaging</h1>

<p align="center">
  <strong>Quoc-Huy Trinh, Minh-Van Nguyen, Jung Zeng, Ulas Bagci, Debesh Jha</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.11872"><img src="https://img.shields.io/badge/arXiv-2505.11872-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huyquoctrinh.github.io/prsmed/"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/huyquoctrinh/PRS-Med"><img src="https://img.shields.io/badge/🤗-Model-orange.svg" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/datasets/huyquoctrinh/PRS-Med"><img src="https://img.shields.io/badge/🤗-Dataset-yellow.svg" alt="HuggingFace Dataset"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.11872">Paper</a> |
  <a href="https://huyquoctrinh.github.io/prsmed/">Project Page</a> |
  <a href="https://huggingface.co/huyquoctrinh/PRS-Med">Models</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#training">Training</a> |
  <a href="#inference">Inference</a> |
  <a href="#evaluation">Evaluation</a>
</p>

---

**PRS-Med** is a modular framework for training and inference of segmentation models powered by large language models (LLMs). It integrates components like LLaVA, Segment Anything, and TinySAM to perform multimodal position reasoning segmentation tasks in medical imaging.

## 🔔 News

- **[2025.09.23]** Paper is accepted at CVPRW 2026!
- **[2025.09.23]** Published the PRS-Med dataset, including Medical Position QA, Multiple Choice QA about position and medical reasoning.
- **[2025.06.01]** Updated repository of PRS-Med.

## 🔧 Features

- Support Position Reasoning Segmentation task
- Support evaluation tool for reasoning
- Support evaluation tool for segmentation
- Support training and inference of the model

---

## <a name="dataset"></a> 📦 Dataset

For the PRS-Med dataset, it is available in:

| Source | Link |
|--------|------|
| Google Drive | [Part 1](https://drive.google.com/file/d/1vY6UD4bfccdIDRpwpG_nVZ9r1vSYPRd1/view?usp=drive_link), [Part 2](https://drive.google.com/file/d/1Lt0y9UiQFDQ9PgnW1oYy1hW6I211Glot/view?usp=drive_link), [Annotations](https://drive.google.com/drive/folders/1VyFqcfDbvrtYBA13ZkDz0scQmehLYYzt?usp=drive_link) |
| Hugging Face | [![Hugging Face](https://img.shields.io/badge/🤗-Dataset%20Repo-yellow.svg)](https://huggingface.co/datasets/huyquoctrinh/PRS-Med) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/huyquoctrinh/PRS-Med.git
cd PRS-Med
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights *(Will be updated soon)*

```bash
bash download.sh
```

---

## <a name="training"></a> 🏋️‍♂️ Training

> Full training code is not completely updated yet.

Use `train.py` to train a segmentation model.

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ddp.py \
  --model_path /path/to/your/based_model \
  --data_path /path/to/your/images/data \
  --annotation_path /path/to/your/annotations \
  --batch_size 4 --epochs 50 --save_dir /path/to/your/save/dir \
  --grad_accum_steps 8 \
  --grad_clip_norm 1.0
```

---

## <a name="inference"></a> 🧪 Inference

Use `infer.py` to perform inference on images.

```bash
python infer_full.py
```

> **Note:** Please update your checkpoint inside that folder to match with your trained model.

---

## <a name="evaluation"></a> 📊 Evaluation *(will be updated later)*

Use the scripts in the `evaluation/` directory to assess model performance. There are two evaluation tools for reasoning and segmentation.

---

## 📈 Visualization *(will be updated later)*

Use `visualize.py` to visualize segmentation results.

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

## 📌 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{trinh2025prs,
  title   = {Prs-med: Position reasoning segmentation with vision-language model in medical imaging},
  author  = {Trinh, Quoc-Huy and Nguyen, Minh-Van and Zeng, Jung and Bagci, Ulas and Jha, Debesh},
  journal = {arXiv preprint arXiv:2505.11872},
  year    = {2025}
}
```

## 🙏 Acknowledgement

This work is built upon [LLaVA](https://github.com/haotian-liu/LLaVA), [Segment Anything](https://github.com/facebookresearch/segment-anything), and [TinySAM](https://github.com/xinghaochen/TinySAM).
