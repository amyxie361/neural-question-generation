EXP=wiki_0.1m_quora_0.1m_batch32
echo "rel"

python eval.py \
    --out_file=../result/${EXP}/generated.txt \
    --src_file=../result/${EXP}/golden.txt \
    --tgt_file=../result/${EXP}/golden.txt

echo "origin"
python eval.py \
    --out_file=../result/${EXP}/generated.txt \
    --src_file=../result/${EXP}/origin.txt \
    --tgt_file=../result/${EXP}/origin.txt
