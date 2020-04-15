# train file
glove_path = "/data/yqxie/00_data/GloVe"
train_src_file = "/data/yqxie/01_exps/102_qg/neural-question-generation/quora_train_src.txt"
train_trg_file = "/data/yqxie/01_exps/102_qg/neural-question-generation/quora_train_trg.txt"
# dev file
dev_src_file = train_src_file
dev_trg_file = train_trg_file
embedding = glove_path + "/embedding_en_wiki.pkl"
word2idx_file = glove_path + "/word2idx_en_wiki.pkl"

exp_name = "quora_batch1"
model_path = "./save/" + exp_name + "/train_414230136/42_0.85"
train = True
test = False
device = "cuda:5"
use_gpu = True
debug = False
debug_num = 100
data_num = 1e5
vocab_size = 45000
freeze_embedding = False
use_size = 512

num_epochs = 1000000
max_seq_len = 100
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 1e-2
decay_start = 8
decay_step = 2
decay_weight = 0.8
batch_size = 1
dropout = 0.3
max_grad_norm = 20.0

beam_size = 8
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/" + exp_name
