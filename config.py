# train file
glove_path = "/data/yqxie/00_data/GloVe"
#train_src_file = "/data/yqxie/01_exps/102_qg/neural-question-generation/quora_train_src.txt"
train_src_file = "/data/yqxie/01_exps/100_paraphrase/paraphrase/train_sent_token.txt"
train_trg_file = train_src_file
# train_trg_file = "/data/yqxie/01_exps/102_qg/neural-question-generation/quora_train_trg.txt"
# dev file
dev_src_file = train_src_file
dev_trg_file = train_trg_file
embedding = glove_path + "/embedding_en_wiki.pkl"
word2idx_file = glove_path + "/word2idx_en_wiki.pkl"

exp_name = "wiki_0.1m_quora_0.1m_batch32"
# exp_name = "quora_10k_batch32"
# exp_name = "quora_1m_batch32"
# exp_name = "quora_0.1m_batch32"
# model_path = "./save/wiki_0.1m_quora_batch32/train_415121702/5_4.47"
model_path = "./save/wiki_0.1m_quora_0.1m_batch32/train_415160304/4_1.07"
exp_name = "wiki_10m_batch32"
load_model = False
train = False
test = False
interface = True
device = "cuda:7"
use_gpu = True
debug = False
debug_num = 100
#data_num = 1e7
data_num = 10
vocab_size = 45000
freeze_embedding = False
use_size = 512

num_epochs = 1000000
max_seq_len = 100
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 3e-4
decay_start = 8
decay_step = 2
decay_weight = 0.8
batch_size = 32
dropout = 0.3
max_grad_norm = 20.0

beam_size = 3
min_decode_step = 5
max_decode_step = 100
output_dir = "./result/" + exp_name
