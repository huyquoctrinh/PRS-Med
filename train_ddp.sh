torchrun --nproc_per_node=3 train_ddp.py \
  --model_path /home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b \
  --data_path /home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data \
  --annotation_path /home/mamba/ML_project/Testing/Huy/llm_seg/dataset/annotation_v3 \
  --batch_size 4 --epochs 20 --save_dir /home/mamba/ML_project/Testing/Huy/llm_seg/training_results/train_sam_med_llava_med_new_fixed
