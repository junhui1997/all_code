cd ..
for e_layers in 2 3 4
do
for top_k in 2 4 6 8 16
do
  for d_model in 16 32 64 128 256
do
  for factor in 1
do
  python -u tcn_main.py --split 8 --save_name hhh --task_name su --e_layers $e_layers  --factor $factor  --d_model $d_model --d_ff 128  --top_k $top_k
done
done
done
done
###