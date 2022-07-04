from os.path import join


TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

MODEL_DIR = 'model-folder'
EMB = 'emb.pt'
ENCODER = 'encoder.pt'
MLM = 'mlm.pt'

MODEL_EMB = join(MODEL_DIR, EMB)
MODEL_ENCODER = join(MODEL_DIR, ENCODER)
MODEL_MLM = join(MODEL_DIR, MLM)

BOARD_NAME = 'bert_cb_news'
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'

EPOCH_NUM = 25

######
#  DEVICE
######

CUDA0 = 'cuda:0'
CUDA1 = 'cuda:1'
CUDA2 = 'cuda:2'
CUDA3 = 'cuda:3'
CPU = 'cpu'

########
#   VOCAB
########

UNK = '<unk>'
PAD = '<pad>'
CLS = '<cls>'
SEP = '<sep>'
MASK = '<mask>'

MAIN_VOCAB = 10005  # TODO Размер основного словаря
UNK_SIZE = 2500  # TODO Количество индексов под неизвестные слова
SEQ_LEN = 512  # Максимальная длина входной последовательности в токенах
EMB_DIM = 768  # Размерность эмбеддинга
LAYERS_NUM = 12  # Количество слоёв в энкодере-трансформере
HEADS_NUM = 12  # Количество голов внимания
HIDDEN_DIM = 3072  # Размерность скрытого слоя
DROPOUT = 0.1
NORM_EPS = 1e-12
