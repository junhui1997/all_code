cd ../..
for model_name in Autoformer  Transformer Nonstationary_Transformer DLinear FEDformer Informer LightTS Reformer PatchTST Pyraformer MICN Crossformer lstm conv_net bp lstm_fcn fcn fcn_m conv_next TimesNet my
do
for seq_len in 250 500
do
  for e_layers in 2
do
  for factor in 2
do
  python -u run.py \
  --task_name classification --enc_in 12   --is_training 1    --root_path ./    --model_id bone_c   --data bone_drill_c    --e_layers $e_layers    --batch_size 16    --d_model 64  --factor $factor  --d_ff 64    --top_k 3    --des 'Exp'    --itr 5    --learning_rate 0.001    --train_epochs 30    --patience 10 --seq_len $seq_len --filter mean --model $model_name
done
done
done
done