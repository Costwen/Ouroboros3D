n_nodes=1
n_gpus_per_node=8 # number of gpus
train_cmd="torchrun --nnodes=$n_nodes --nproc_per_node=$n_gpus_per_node"

function train(){
    $train_cmd train.py --config configs/mv/svd_lgm_rgb_ccm_lvis.yaml
}

function inference(){
    python inference.py \
        --input testset/3d_arena_a_black_t-shirt_with_the_peace_sign_on_it.png \
        --checkpoint /path/to/checkpoint \
        --output workspace \
        --seed 42 \
        --config configs/mv/infer.yaml
}

$1