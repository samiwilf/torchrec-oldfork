# DLRM run command:
# torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py
# torchx run -s local_cwd  aws_component.py:run_dlrm_main --num_trainers=8

import os
rank = int(os.environ["LOCAL_RANK"])

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

import pathlib

LOG_PATH = "/home/ubuntu/repos/torchrec-oldfork/examples/dlrm/"
SETTING = 5

if rank == 0:
    print("\n")
    print('*'.center(80, '*'))
    print(f"  RUNNING SETTING {SETTING}  ".center(80, '*'))
    #print('*'.center(80, '*'))
    print(f"  GIT HASH {sha}  ".center(80, '*'))
    print('*'.center(80, '*'))
    print("\n")
    for diff_item in repo.index.diff(None):
        # print(diff_item.new_file)
        if diff_item.b_path[-3:] == ".py":
            #print(diff_item.b_path)
            diff = repo.git.diff(repo.head.commit.tree, diff_item.b_path)
            print(diff)
    print("\n")

DENSE_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_DENSE.txt")
SPARSE_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_SPARSE.txt")
D_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_D_OUT.txt")
E_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_E_OUT.txt")
C_OUT_LOG_FILE = pathlib.Path(LOG_PATH + "s" + str(SETTING) + "_C_OUT.txt")


if SETTING == 1:
    MODEL_EVAL = False
    SAVE_DEBUG_DATA = False
    SAVE_LOSSES = True
    NP_WEIGHT_INIT = True
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
    MODEL_EVAL = False
    SAVE_DEBUG_DATA = False
    SAVE_LOSSES = True
    NP_WEIGHT_INIT = True
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
    MODEL_EVAL = False
    SAVE_DEBUG_DATA = False
    SAVE_LOSSES = True
    NP_WEIGHT_INIT = True
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
    MODEL_EVAL = True
    SAVE_DEBUG_DATA = False
    SAVE_LOSSES = False
    NP_WEIGHT_INIT = False
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
    ]

if SETTING == 5:
    NEW_DATASET_SHUFFLED = True
    NEW_DATASET = True
    MLPERF = False
    MODEL_EVAL = True
    SAVE_DEBUG_DATA = False
    SAVE_LOSSES = False
    NP_WEIGHT_INIT = False
    LOG_FILE = "losses5.txt"
    INT_FEATURE_COUNT = 13
    #CAT_FEATURE_COUNT = 26
    DAYS = 24
    BATCH_SIZE = 2048 #16384 # 256
    EMB_DIM = 128

    #mlperf
    if MLPERF:
        LN_EMB = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36]
    else:
        #maxim's version
        LN_EMB=[45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35]

    import math
    POWER_NUM = 1
    RESCALE_THRESHOLD = 10000000000
    LN_EMB_RESCALED = [0 if x < RESCALE_THRESHOLD else 1 for x in LN_EMB]
    LN_EMB = [x if x < RESCALE_THRESHOLD else math.ceil(x**POWER_NUM) for x in LN_EMB]

    ARGV = [
        '--embedding_dim', f'{EMB_DIM}',
        '--dense_arch_layer_sizes', f'512,256,{EMB_DIM}',
        '--over_arch_layer_sizes', '1024,1024,512,256,1',
    ]
    ARGV += (['--shuffle_batches', 'True', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/shuffled',]
        if NEW_DATASET_SHUFFLED
        else  ['--shuffle_batches', 'True', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/1tb_numpy',]
        if NEW_DATASET
        else ['--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M',])


LOG_FILE = pathlib.Path(LOG_PATH + LOG_FILE)
for f in [LOG_FILE, DENSE_LOG_FILE, SPARSE_LOG_FILE, D_OUT_LOG_FILE, E_OUT_LOG_FILE, C_OUT_LOG_FILE]:
    try:
        f.unlink()
    except:
        pass
    # try:
    #     import shutil
    #     dir_path = 'tensorboard_file'
    #     shutil.rmtree(dir_path)
    # except:
    #     pass

import time
COMMON_ARGV = [
    '--batch_size', str(BATCH_SIZE),
    '--num_embeddings_per_feature', ','.join([str(x) for x in LN_EMB]),
    '--epochs', '1',
    # '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M',
    '--pin_memory',
    '--learning_rate', '1.0',
    '--num_workers', '4',
    #'--validation_freq_within_epoch','30000',
    '--validation_freq_within_epoch','100000',
    #'--mlperf_logging',
    '--multi_hot_size','20',
    # '--collect_freqs_stats',
    #'--tensor_board_filename', '1-hot-SGD-LR1-TB-data-table-wise-batched-fused-256-batch-size-' + str(int(time.time())),
    #'--tensor_board_filename', '20-hot-attention-SGD-LR1-Uniform-TB-data-' + str(int(time.time())),
    # '--interaction_branch1_layer_sizes', '128,512,512,4096',
    # '--interaction_branch2_layer_sizes', '512,512,4096',
    '--tensor_board_filename', '20-softmax-SGD-LR1-Uniform-2048-batch-TB-data-' + str(int(time.time())),
]
# COMMON_ARGV += ['--limit_train_batches','100', '--limit_val_batches', '100', '--limit_test_batches', '100']
if SETTING != 5 and SETTING != 4:
    COMMON_ARGV += ['--limit_train_batches', '50', '--limit_val_batches', '5', '--limit_test_batches', '5']
#COMMON_ARGV += ['--limit_train_batches', '256055', '--limit_val_batches', '5441', '--limit_test_batches', '5431']
# dlrm oss's number is 2048437 instead of 256055 when using 8 gpus. And  2048437 / 256055 ~= 8

ARGV = ARGV + COMMON_ARGV

CAT_FEATURE_COUNT = len(ARGV[(ARGV.index('--num_embeddings_per_feature')+1)].split(','))

#86 it/s for 1-hot
#28 it/s for 20-hot