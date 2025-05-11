#!/bin/bash
#SBATCH --job-name=SPIN_ODE
#SBATCH --output=soutput/SPIN_ODE/%j.out
#SBATCH --account=project_2009907
#SBATCH --partition=gpusmall
#SBATCH --time=02:15:00
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1

##SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:a100_1g.5gb:1


## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

echo "===== SLURM SCRIPT  ====="
cat $0
echo "==================================="

module load jax
source /scratch/project_2009907/nn_reaction_rate/venv_jax/bin/activate
alias python="srun /scratch/project_2009907/nn_reaction_rate/venv_jax/bin/python"


# proposed approach
## step 1: train MLP to fit nODE traj
# python nnrr-jax.py --config configs/spin.yaml --target rober_fit
# python nnrr-jax.py --config configs/spin.yaml --target pollu_fit
# python nnrr-jax.py --config configs/spin.yaml --target toy_fit

## step 2: train CRNN with deriv from interpolated traj inferenced by MLP
# python nnrr-jax-coll.py --config configs/spin.yaml --target rober_coll
# python nnrr-jax-coll.py --config configs/spin.yaml --target pollu_coll
# python nnrr-jax-coll.py --config configs/spin.yaml --target toy_coll

## step : fine-tune on CRNN with estimated rate coefficient
# python nnrr-jax.py --config configs/spin.yaml --target rober_tune
# python nnrr-jax.py --config configs/spin.yaml --target pollu_tune
# python nnrr-jax.py --config configs/spin.yaml --target toy_tune

##################################
# Baseline: directly fit CRNN on traj with ODESolver
# python nnrr-jax.py --config configs/crnn_ode.yaml --target rober
# python nnrr-jax.py --config configs/crnn_ode.yaml --target pollu
# python nnrr-jax.py --config configs/crnn_ode.yaml --target toy

##################################
# Ablation
# w/o diff: train CRNN with deriv learned from MLP
# python nnrr-jax-coll.py --config configs/coll_mlpoutput.yaml --target rober_coll
# python nnrr-jax-coll.py --config configs/coll_mlpoutput.yaml --target pollu_coll
# python nnrr-jax-coll.py --config configs/coll_mlpoutput.yaml --target toy_coll

# w/o interpolation: train CRNN with deriv from origin traj
# python nnrr-jax-coll.py --config configs/coll_difforigin.yaml --target rober
# python nnrr-jax-coll.py --config configs/coll_difforigin.yaml --target pollu
# python nnrr-jax-coll.py --config configs/coll_difforigin.yaml --target toy

# w/o physical loss
## step 1: train MLP to fit nODE traj
# python nnrr-jax.py --config configs/spin_phyloss.yaml --target rober_fit
# python nnrr-jax.py --config configs/spin_phyloss.yaml --target pollu_fit
# python nnrr-jax.py --config configs/spin_phyloss.yaml --target toy_fit

## step 2: train CRNN with deriv from interpolated traj inferenced by MLP
# python nnrr-jax-coll.py --config configs/spin_phyloss.yaml --target rober_coll
# python nnrr-jax-coll.py --config configs/spin_phyloss.yaml --target pollu_coll
# python nnrr-jax-coll.py --config configs/spin_phyloss.yaml --target toy_coll

# Stiffness

# Toy sample points
# python nnrr-jax.py --config configs/spin_reduce.yaml --target rober_fit
# python nnrr-jax.py --config configs/spin_reduce.yaml --target pollu_fit
# python nnrr-jax.py --config configs/spin_reduce.yaml --target toy_fit

# python nnrr-jax-coll.py --config configs/spin_reduce.yaml --target rober_coll
# python nnrr-jax-coll.py --config configs/spin_reduce.yaml --target pollu_coll
# python nnrr-jax-coll.py --config configs/spin_reduce.yaml --target toy_coll