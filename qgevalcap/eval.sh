EXP=new_data2_80k_noFreeze_train
python eval.py \
    --out_file=../result/${EXP}/generated.txt \
    --src_file=../result/${EXP}/golden.txt \
    --tgt_file=../result/${EXP}/golden.txt
