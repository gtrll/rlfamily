#!/bin/bash
# x lower, x upper, y lower, y upper
set_variables_helper() {
    N_ITERS=$1
    Y_LOWER=$2
    Y_HIGHER=$3
    LEGEND_LOC=$4
}
set_variables() {
    if [ $1 == cp ]; then
        set_variables_helper 100 0 1000 4
    elif [ $1 == hopper ]; then
        set_variables_helper 200 0 5700 0
    elif [ $1 == snake ]; then
        set_variables_helper 200 0 4200 0
    elif [ $1 == walker3d ]; then
        set_variables_helper 1000 0 3000 2
    else
        echo "Error: environment not recognized: $1"
        exit 1
    fi
}

# The organization of folder is
# root
#     env_alg  (e.g. hopper_adam)
#         oracle  (e.g. model-free, last, which are renamed to manually)
DIR='/absolute/path/to/experiments/root/folder'
# General figures.
if [ $1 = "general" ] || [ $1 = "all" ]; then
    declare -a ENVS=("cp" "hopper" "snake" "walker3d")
    declare -a ALGS=("adam" "natgrad" "trpo")
    for ENV in "${ENVS[@]}"; do
        set_variables $ENV
        for ALG in "${ALGS[@]}"; do
            python scripts/plot.py --logdir_parent ${DIR}/${ENV}_${ALG} --value MeanSumOfRewards --style icml_piccolo_final --output ../${ENV}_${ALG}.pdf --n_iters $N_ITERS --y_lower $Y_LOWER --y_higher $Y_HIGHER --legend_loc $LEGEND_LOC &
        done
    done
fi

if [ $1 = "models" ] || [ $1 = "all" ]; then
    # Cartpole different models fidelities.
    declare -a ENVS=("cp")
    declare -a ALGS=("adam" "natgrad" "trpo")
    for ENV in "${ENVS[@]}"; do
        set_variables $ENV
        for ALG in "${ALGS[@]}"; do
            python scripts/plot.py --logdir_parent ${DIR}/${ENV}_${ALG}_models --value MeanSumOfRewards --style icml_piccolo_final --output ../${ENV}_${ALG}_models.pdf --n_iters $N_ITERS --y_lower $Y_LOWER --y_higher $Y_HIGHER --legend_loc $LEGEND_LOC &            
        done
    done    
fi

if [ $1 = "adv" ] || [ $1 = "all" ]; then
    # Cartpole different models fidelities.
    declare -a ENVS=("cp")
    declare -a ALGS=("adam" "natgrad" "trpo")
    for ENV in "${ENVS[@]}"; do
        set_variables $ENV
        # Need more iterations.
        for ALG in "${ALGS[@]}"; do
            python scripts/plot.py --logdir_parent ${DIR}/${ENV}_${ALG}_adv --value MeanSumOfRewards --style icml_piccolo_final --output ../${ENV}_${ALG}_adv.pdf --n_iters $N_ITERS --y_lower $Y_LOWER --y_higher $Y_HIGHER --legend_loc $LEGEND_LOC &            
        done
    done        

    
fi
