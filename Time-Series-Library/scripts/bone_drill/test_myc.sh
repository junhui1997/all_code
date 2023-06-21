cd ../..
for model_name in my
do
for seq_len in 500
do
for moving_avg in 5 25 50 100
do
  for e_layers in 1 2
do
  for d_layers in 1
do
  for factor in 1 2 3
do
  python -u run.py \
  --task_name classification --enc_in 12   --is_training 1    --root_path ./    --model_id bone_c   --data bone_drill_c    --e_layers $e_layers  --d_layers $d_layers    --batch_size 16    --d_model 64  --factor $factor  --d_ff 64    --top_k 3    --des 'Exp'    --itr 25    --learning_rate 0.0001    --train_epochs 40    --patience 15 --seq_len $seq_len --filter mean --model $model_name
done
done
done
done
done
done