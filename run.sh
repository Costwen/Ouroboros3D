n_nodes=1
n_gpus_per_node=8 # number of gpus
train_cmd="torchrun --nnodes=$n_nodes --nproc_per_node=$n_gpus_per_node"

function train(){
    $train_cmd train.py --config configs/mv/svd_lgm_rgb_ccm_lvis.yaml
}

function infer(){
    python infer.py --config configs/mv/infer.yaml --input testset/knife.png
}

$1