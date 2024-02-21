#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

#python src/vanderpol.py --train_type "uniform" --test_type "uniform" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "uniform" --test_type "trajectory" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "uniform" --test_type "trajectory_ls" --use_approximate_model True --grad_steps 1000 --plot_time 500

#python src/vanderpol.py --train_type "trajectory" --test_type "trajectory" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "trajectory" --test_type "trajectory_ls" --use_approximate_model True --grad_steps 1000 --plot_time 500


# train policies
#python src/train_policy.py --env "Ant-v3" --num-timesteps 10000000 --alg sac
#python src/train_policy.py --env "HalfCheetah-v3" --num-timesteps 10000000 --alg sac



# Note these can all be run in parallel on the cpu.
# Sleeps prevent race conditions
#python src/gather_trajectories.py --alg sac --env "HalfCheetah-v3" --num_envs 200 --transitions_per_env 50000 --data_type "on-policy" &
#sleep 2
#python src/gather_trajectories.py --alg sac --env "HalfCheetah-v3" --num_envs 200 --transitions_per_env 50000 --data_type "random" &
#sleep 2
#python src/gather_trajectories.py --alg sac --env "HalfCheetah-v3" --num_envs 200 --transitions_per_env 50000 --data_type "precise" &
#sleep 2
#python src/gather_trajectories.py --alg sac --env "HalfCheetah-v3" --num_envs 200 --transitions_per_env 50000 --data_type "precise2" &
#wait


#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "on-policy"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "on-policy"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "on-policy"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "on-policy"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "on-policy"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "on-policy"

python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "random" --normalize
python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "random" --normalize
python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "random" --normalize
python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "random" --normalize
python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "random" --normalize
python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize

#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "precise"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "precise"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "precise"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "precise"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "precise"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "precise"
#
#
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "precise2"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "precise2"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "precise2"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "precise2"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "precise2"
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "precise2"