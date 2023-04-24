cd ../..
#python -u run.py    --task_name long_term_forecast    --is_training 1    --root_path ././dataset/    --data_path ETTm1.csv   --model_id bone_drill_192_720mean    --model TimesNet    --data bone_drill    --features M    --seq_len 192    --label_len 192    --pred_len 720    --e_layers 2    --d_layers 1    --factor 3    --enc_in 6    --dec_in 6    --c_out 6    --d_model 16    --d_ff 32    --des 'Exp'    --itr 1    --top_k 5 --embed none --filter low_pass --train_epochs 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ././dataset/  \
  --data_path ETTh1.csv \
  --model_id bone_drill_192_720mean \
  --model Autoformer\
  --data bone_drill \
  --features M \
  --seq_len 192 \
  --label_len 192 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --embed none \
  --itr 1