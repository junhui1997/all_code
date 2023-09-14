cd ../..
for model_name in  Transformer Nonstationary_Transformer
do
for seq_len in 1000
do
for moving_avg in 100
do
  for e_layers in 2
do
  for d_layers in 1
do
  for factor in 1
do
  for filter in mean
do
  python -u run.py \
  --task_name classification --enc_in 12   --is_training 1    --root_path ./  --moving_avg $moving_avg  --model_id bone_c   --data bone_drill_c    --e_layers $e_layers  --d_layers $d_layers    --batch_size 16    --d_model 64  --factor $factor  --d_ff 64    --top_k 3    --des 'Exp'    --itr 15    --learning_rate 0.0001    --train_epochs 30    --patience 10 --seq_len $seq_len --filter $filter --model $model_name
done
done
done
done
done
done
done