cd ../..
python -u run.py   --task_name classification --enc_in 6   --is_training 1    --root_path ./    --model_id bone_c    --model TimesNet    --data bone_drill_c    --e_layers 2    --batch_size 16    --d_model 64    --d_ff 64    --top_k 3    --des 'Exp'    --itr 1    --learning_rate 0.001    --train_epochs 30    --patience 10 --seq_len 400 --filter low_pass