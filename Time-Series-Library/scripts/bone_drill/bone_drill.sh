cd ../..
python -u run.py    --task_name long_term_forecast    --is_training 1    --root_path ././dataset/    --data_path ETTm1.csv   --model_id bone_drill_96_32    --model TimesNet    --data bone_drill    --features M    --seq_len 96    --label_len 48    --pred_len 96    --e_layers 1    --d_layers 1    --factor 3    --enc_in 6    --dec_in 6    --c_out 6    --d_model 16    --d_ff 32    --des 'Exp'    --itr 1    --top_k 5 --embed none --filter low_pass --train_epochs 10