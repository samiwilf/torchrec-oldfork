# DLRM run command: 
# torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py
# torchx run -s local_cwd  aws_component.py:run_dlrm_main --num_trainers=8

DATASET = "1tbnumpy"
MLP_INIT_TYPE = "normal"
TRI_TYPE = "top"



# choices:
DATASETS = ["subsample0", "1tbnumpy"]
TRI_TYPES = ["bottom", "top"]
MLP_INIT_TYPES = ["uniform", "normal"]

if DATASET == "subsample0":
    DATASET_PATH = "/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M"
else:
    DATASET_PATH = "/home/ubuntu/mountpoint/1tb_numpy"
