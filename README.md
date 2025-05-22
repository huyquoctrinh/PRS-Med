# PRS-Med

**PRS-Med** is a modular framework for training and inference of segmentation models powered by large language models (LLMs). It integrates components like LLaVA, Segment Anything, and TinySAM to perform multimodal segmentation tasks.

## 🔧 Features

* Train custom segmentation models with `train.py`.
* Perform inference on images using `infer.py`.
* Supports integration with LLaVA, Segment Anything, and TinySAM.
* Includes utilities for data processing, evaluation, and visualization.([GitHub][1])

---

## 📁 Repository Structure

```

LLM_segmentation/
├── data_processing/         # Scripts for data preprocessing
├── data_utils/              # Utilities for data handling
├── evaluation/              # Evaluation metrics and scripts
├── llava/                   # LLaVA integration
├── logs/                    # Training and inference logs
├── segment_anything/        # Segment Anything model integration
├── segment_model/           # Core segmentation models
├── tinysam/                 # TinySAM model integration
├── weights3_norm/           # Pretrained weights
├── train.py                 # Training script
├── infer.py                 # Inference script
├── requirements.txt         # Python dependencies
└── ...                      # Additional scripts and assets
```



---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/huyquoctrinh/LLM_segmentation.git
cd LLM_segmentation
```



### 2. Install Dependencies

```bash
pip install -r requirements.txt
```



### 3. Download Pretrained Weights (Optional)

```bash
bash download.sh
```



---

## 🏋️‍♂️ Training

Use `train.py` to train a segmentation model.

### Example:

```bash
python train.py \
  --data_dir path/to/dataset \
  --model_name segment_model \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --output_dir weights/
```



### Arguments:

* `--data_dir`: Path to the training dataset.
* `--model_name`: Name of the model to train (e.g., `segment_model`).
* `--epochs`: Number of training epochs.
* `--batch_size`: Batch size for training.
* `--learning_rate`: Learning rate for the optimizer.
* `--output_dir`: Directory to save trained weights.

---

## 🧪 Inference

Use `infer.py` to perform inference on images.

### Example:

```bash
python infer.py \
  --weights weights/model.pth \
  --input_image path/to/image.jpg \
  --output_mask path/to/output_mask.png \
  --visualize
```



### Arguments:

* `--weights`: Path to the trained model weights.
* `--input_image`: Path to the input image.
* `--output_mask`: Path to save the output mask.
* `--visualize`: Flag to visualize the input image and output mask.([GitHub][2])

---

## 📊 Evaluation

Use the scripts in the `evaluation/` directory to assess model performance.

### Example:

```bash
python evaluation/evaluate.py \
  --predictions_dir path/to/predictions \
  --ground_truth_dir path/to/ground_truth
```



---

## 📈 Visualization

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

[1]: https://github.com/deep-diver/LLM-Serve?utm_source=chatgpt.com "GitHub - deep-diver/LLM-Serve: This repository provides a framework to ..."
[2]: https://github.com/yakhyo/crack-segmentation/blob/main/inference.py?utm_source=chatgpt.com "crack-segmentation/inference.py at main - GitHub"
