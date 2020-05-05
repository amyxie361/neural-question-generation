EXP=new_data2_80k_noFreeze_train
PAT=../result/${EXP}
PAT=../../t5-colab-results/squad/
python eval.py \
    --out_file=${PAT}/predict.txt \
    --src_file=${PAT}/golden.txt \
    --tgt_file=${PAT}/golden.txt
