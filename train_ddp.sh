CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ddp.py \
  --model_path /home/mamba/ML_project/Testing/Huy/llm_seg/weight/llava-med-v1.5-mistral-7b \
  --data_path /home/mamba/ML_project/Testing/Huy/llm_seg/dataset/prs_med/data \
  --annotation_path /home/mamba/ML_project/Testing/Huy/llm_seg/dataset/prs_med/annotations \
  --batch_size 4 \
  --epochs 20 \
  --save_dir /home/mamba/ML_project/Testing/Huy/llm_seg/training_results/train_prs_med_clean \
  --grad_accum_steps 8 \
  --grad_clip_norm 1.0
