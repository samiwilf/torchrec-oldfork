# DLRM run command: 
# torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_main.py
# torchx run -s local_cwd  aws_component.py:run_dlrm_main --num_trainers=8

import pathlib

LOG_PATH = "/home/ubuntu/repos/torchrec-fork/examples/dlrm/"
SETTING = 3
print('*'.center(40, '*'))
print(f"  RUNNING SETTING {SETTING}  ".center(40, '*'))
print('*'.center(40, '*'))

#Defaults, may be overwritten by settings below.
MODEL_EVAL = False
SAVE_DEBUG_DATA = True

DENSE_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_DENSE.txt")
SPARSE_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_SPARSE.txt")
D_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_D_OUT.txt")
E_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_E_OUT.txt")
C_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_C_OUT.txt")

if SETTING == 1:
    LOG_FILE = "s1_losses_simplestNN.txt"    
    INT_FEATURE_COUNT = 1
    DAYS = 1
    BATCH_SIZE = 1
    EMB_DIM = 4
    LN_EMB=[1] 
    ARGV = [  
        '--embedding_dim', f'{EMB_DIM}', 
        '--dense_arch_layer_sizes', f'{EMB_DIM}', 
        '--over_arch_layer_sizes', '1,1', 
    ]

if SETTING == 2:
    LOG_FILE = "s2_losses_simplestNN.txt"    
    INT_FEATURE_COUNT = 1
    DAYS = 1
    BATCH_SIZE = 8388608
    EMB_DIM = 4
    LN_EMB=[1]
    ARGV = [ 
        '--embedding_dim', f'{EMB_DIM}', 
        '--dense_arch_layer_sizes', f'{EMB_DIM}', 
        '--over_arch_layer_sizes', '1,1', 
    ]

if SETTING == 3:
    SAVE_DEBUG_DATA = False
    LOG_FILE = "s3.txt" 
    INT_FEATURE_COUNT = 1 
    DAYS = 1
    BATCH_SIZE = 2048
    EMB_DIM = 128
    #LN_EMB=[16,21,34,18,13]
    LN_EMB=[4538,346,175]
    ARGV = [ 
        '--embedding_dim', f'{EMB_DIM}', 
        '--dense_arch_layer_sizes', f'512,256,{EMB_DIM}', 
        '--over_arch_layer_sizes', '1024,1024,512,256,1', 
    ]

if SETTING == 4:
    SAVE_DEBUG_DATA = False
    LOG_FILE = "s4.txt"
    INT_FEATURE_COUNT = 13
    DAYS = 1
    BATCH_SIZE = 2048 
    EMB_DIM = 128
    LN_EMB=[45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35]
    ARGV = [ 
        '--embedding_dim', f'{EMB_DIM}', 
        '--dense_arch_layer_sizes', f'512,256,{EMB_DIM}', 
        '--over_arch_layer_sizes', '1024,1024,512,256,1', 
        "--num_embeddings", '40000000',
    ]

if SETTING == 5:
    SAVE_DEBUG_DATA = False
    MODEL_EVAL = True
    LOG_FILE = "s5_losses_terabyte_full.txt"
    INT_FEATURE_COUNT = 13
    #CAT_FEATURE_COUNT = 26 
    DAYS = 24
    BATCH_SIZE = 2048
    EMB_DIM = 128
    LN_EMB=[45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35]
    ARGV = [ 
        '--embedding_dim', f'{EMB_DIM}', 
        '--dense_arch_layer_sizes', f'512,256,{EMB_DIM}', 
        '--over_arch_layer_sizes', '1024,1024,512,256,1',
        "--num_embeddings", '40000000',
    ]

LOG_FILE = pathlib.Path(LOG_PATH + LOG_FILE)
for f in [LOG_FILE, DENSE_LOG_FILE, SPARSE_LOG_FILE, D_OUT_LOG_FILE, E_OUT_LOG_FILE, C_OUT_LOG_FILE]:
    if f.is_file():
        f.unlink()

COMMON_ARGV = [
    '--batch_size', str(BATCH_SIZE),
    '--num_embeddings_per_feature', ','.join([str(x) for x in LN_EMB]),
    '--epochs', '1',
    '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M',
    '--pin_memory',
    '--learning_rate', '1.0',
]

if SETTING != 5 and SETTING != 4:
    COMMON_ARGV += ['--limit_train_batches', '50'] #, '--limit_val_batches', '5', '--limit_test_batches', '5']

ARGV = ARGV + COMMON_ARGV

CAT_FEATURE_COUNT = len(ARGV[(ARGV.index('--num_embeddings_per_feature')+1)].split(','))