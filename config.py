# train file
glove_path = "/data/yqxie/00_data/GloVe"
train_src_file = "/data/yqxie/01_exps/100_paraphrase/paraphrase/train_sent_token.txt"
# dev file
dev_src_file = train_src_file
embedding = glove_path + "/embedding_en_wiki.pkl"
word2idx_file = glove_path + "/word2idx_en_wiki.pkl"

exp_name = "paraphrase_clean_debug"
model_path = "./save/" + exp_name + "/train_414204121/612_0.0"
train = False
test = True
device = "cuda:7"
use_gpu = True
debug = True
debug_num = 100
data_num = 1e5
vocab_size = 45000
freeze_embedding = True
use_size = 512

num_epochs = 1000000
max_seq_len = 100
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 3e-3
decay_start = 8
decay_step = 2
decay_weight = 0.8
batch_size = 32
dropout = 0.3
max_grad_norm = 20.0

beam_size = 8
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/" + exp_name
