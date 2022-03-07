# DLRM run command: 
# torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_main.py

LOG_PATH = "/home/ubuntu/repos/torchrec-fork/examples/dlrm/"
SETTING = 1
print(f"***********************************")
print(f"     RUNNING SETTING {SETTING}     ")
print(f"***********************************")
if SETTING == 1:
    LOG_FILE = "s1_losses_simplestNN.txt"    
    INT_FEATURE_COUNT = 1
    CAT_FEATURE_COUNT = 1
    DAYS = 1
    BATCH_SIZE = 1
    ARGV = [  
        '--num_embeddings_per_feature', '1', 
        '--embedding_dim', '4', 
        '--dense_arch_layer_sizes', '4', 
        '--over_arch_layer_sizes', '1,1', 
    ]

if SETTING == 2:
    LOG_FILE = "s2_losses_simplestNN.txt"    
    INT_FEATURE_COUNT = 1
    CAT_FEATURE_COUNT = 1
    DAYS = 1
    BATCH_SIZE = 8388608
    ARGV = [ 
        '--batch_size', str(BATCH_SIZE),  
        '--num_embeddings_per_feature', '1', 
        '--embedding_dim', '4', 
        '--dense_arch_layer_sizes', '4', 
        '--over_arch_layer_sizes', '1,1', 
    ]

if SETTING == 3:
    LOG_FILE = "s3_losses_simplestNN.txt" 
    INT_FEATURE_COUNT = 1 
    CAT_FEATURE_COUNT = 4 
    DAYS = 1
    BATCH_SIZE = 128 
    ARGV = [ 
        '--batch_size', str(BATCH_SIZE),  
        '--num_embeddings_per_feature', '16,16,16,16', 
        '--embedding_dim', '4', 
        '--dense_arch_layer_sizes', '4', 
        '--over_arch_layer_sizes', '1,1', 
    ]

if SETTING == 4:
    LOG_FILE = "s4_losses_day_0_single_sample_.txt"
    INT_FEATURE_COUNT = 13 #1 #13
    CAT_FEATURE_COUNT = 26 #2
    DAYS = 1#24
    BATCH_SIZE = 2048 
    ARGV = [ 
        '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', 
        '--embedding_dim', '128', 
        '--dense_arch_layer_sizes', '512,256,128', 
        '--over_arch_layer_sizes', '1024,1024,512,256,1', 
    ]

if SETTING == 5:
    LOG_FILE = "s5_losses_day_0_single_sample_.txt"
    INT_FEATURE_COUNT = 13 #1 #13
    CAT_FEATURE_COUNT = 26 #2
    DAYS = 1#24
    BATCH_SIZE = 2048 
    ARGV = [ 
        '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', 
        '--embedding_dim', '128', 
        '--dense_arch_layer_sizes', '512,256,128', 
        '--over_arch_layer_sizes', '1024,1024,512,256,1', 
    ]

import pathlib
LOG_FILE = pathlib.Path(LOG_PATH + LOG_FILE)
if LOG_FILE.is_file():
    LOG_FILE.unlink()

COMMON_ARGV = [
    '--batch_size', str(BATCH_SIZE),
    '--epochs', '1',
    '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M',
    '--pin_memory',
    '--learning_rate', '1.0',
    '--limit_train_batches', '200', '--limit_val_batches', '2', '--limit_test_batches', '2',
]

ARGV = ARGV + COMMON_ARGV

    # simplest NN test
    #argv = ['--pin_memory', '--batch_size', '1', '--epochs', '1', '--num_embeddings_per_feature', '1', '--embedding_dim', '4', '--dense_arch_layer_sizes', '4', '--over_arch_layer_sizes', '1,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']

    # simplest 2 GPU NN test
    #argv = ['--pin_memory', '--batch_size', '2', '--epochs', '1', '--num_embeddings_per_feature', '1,1', '--embedding_dim', '4', '--dense_arch_layer_sizes', '4', '--over_arch_layer_sizes', '1,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']

    # real embedding sizes, but fake data and small mlp layers.
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '128', '--over_arch_layer_sizes', '1,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '0.01']

    # real run except limiting test, val, and train batches
    #argv = ['--seed','1','--limit_test_batches', '1', '--limit_val_batches', '1', '--limit_train_batches', '5', '--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']
    
    # real run
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']
    
    # modify embedding tables to be 1 vector per table.
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M', '--learning_rate', '1.0']

    # real run, but using Shabab's 1tb_numpy data
    #argv = ['--pin_memory', '--batch_size', '2048', '--epochs', '1', '--num_embeddings_per_feature', '45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35', '--embedding_dim', '128', '--dense_arch_layer_sizes', '512,256,128', '--over_arch_layer_sizes', '1024,1024,512,256,1', '--in_memory_binary_criteo_path', '/home/ubuntu/mountpoint/criteo/1tb_numpy/', '--learning_rate', '1.0']
