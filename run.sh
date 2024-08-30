n_nodes=1
n_gpus_per_node=8 # number of gpus
train_cmd="torchrun --nnodes=$n_nodes --nproc_per_node=$n_gpus_per_node"
infer_cmd="torchrun --nnodes=1 --nproc_per_node=1"
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH


function train(){
    $train_cmd python train.py --config configs/mv/svd_lgm_rgb_ccm_lvis.yaml
}

function infer(){
    $infer_cmd python infer.py --config configs/mv/infer.yaml --input testset/knife.png
}

$1