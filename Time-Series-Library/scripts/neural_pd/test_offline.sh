cd ../..
python -u run.py --task_name long_term_forecast --lradj type5 --is_training 1 --root_path ././dataset/ --model_id neuralpd --data neural_pd --features M --seq_len 20 --label_len 20 --pred_len 1 --e_layers 2 --d_layers 1 --factor 3 --learning_rate 0.0001 --enc_in 4 --dec_in 2 --c_out 2 --d_model 72 --d_ff 32 --des 'Exp' --itr 1 --top_k 5 --model TimesNet