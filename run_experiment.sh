#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

#python src/vanderpol.py --train_type "uniform" --test_type "uniform" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "uniform" --test_type "trajectory" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "uniform" --test_type "trajectory_ls" --use_approximate_model True --grad_steps 1000 --plot_time 500

#python src/vanderpol.py --train_type "trajectory" --test_type "trajectory" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "trajectory" --test_type "trajectory_ls" --use_approximate_model True --grad_steps 1000 --plot_time 500


# train policies
#python src/train_policy.py --env "Ant-v3" --num-timesteps 10 --alg sac # note i dont use these anymore, but they create directories so we run them
#python src/train_policy.py --env "HalfCheetah-v3" --num-timesteps 10 --alg sac


#python src/gather_trajectories.py --alg sac --env "HalfCheetah-v3" --num_envs 200 --transitions_per_env 50000 --data_type "random" &

#python src/train_predictors.py --env "HalfCheetah-v3" --predictor Oracle --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 0
#
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor Oracle --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 1
#
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor Oracle --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 2
#
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor Oracle --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 3
#
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor Oracle --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor MLP --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor NeuralODE --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "HalfCheetah-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 4

#python src/evaluate_predictors.py --env "HalfCheetah-v3" --data_type "random" --normalize

#python src/gather_trajectories.py --alg sac --env "Ant-v3" --num_envs 200 --transitions_per_env 50000 --data_type "random"

#python src/train_predictors.py --env "Ant-v3" --predictor Oracle --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "Ant-v3" --predictor MLP --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "Ant-v3" --predictor NeuralODE --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "Ant-v3" --predictor FE --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "Ant-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 0
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 0

#python src/train_predictors.py --env "Ant-v3" --predictor Oracle --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "Ant-v3" --predictor MLP --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "Ant-v3" --predictor NeuralODE --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "Ant-v3" --predictor FE --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "Ant-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 1
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 1
#
#python src/train_predictors.py --env "Ant-v3" --predictor Oracle --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "Ant-v3" --predictor MLP --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "Ant-v3" --predictor NeuralODE --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "Ant-v3" --predictor FE --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "Ant-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 2
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 2
#
#python src/train_predictors.py --env "Ant-v3" --predictor Oracle --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "Ant-v3" --predictor MLP --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "Ant-v3" --predictor NeuralODE --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "Ant-v3" --predictor FE --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "Ant-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 3
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 3
#
#python src/train_predictors.py --env "Ant-v3" --predictor Oracle --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "Ant-v3" --predictor MLP --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "Ant-v3" --predictor NeuralODE --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "Ant-v3" --predictor FE --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "Ant-v3" --predictor FE_Residuals --data_type "random" --normalize --seed 4
#python src/train_predictors.py --env "Ant-v3" --predictor FE_NeuralODE_Residuals --data_type "random" --normalize --seed 4
#
#python src/evaluate_predictors.py --env "Ant-v3" --data_type "random" --normalize


#python src/train_policy.py --env "drone" --num-timesteps 10 --alg sac # note i dont use these anymore, but they create directories so we run them
#python src/gather_trajectories.py --alg sac --env "drone" --num_envs 200 --transitions_per_env 50000 --data_type "random"

# python src/train_predictors.py --env "drone" --predictor Oracle --data_type "random" --seed 0 --steps 5000 --normalize
# python src/train_predictors.py --env "drone" --predictor MLP --data_type "random"  --seed 0 --steps 5000 --normalize
# python src/train_predictors.py --env "drone" --predictor NeuralODE --data_type "random"  --seed 0 --steps 5000  --normalize
# python src/train_predictors.py --env "drone" --predictor FE --data_type "random"  --seed 0 --steps 5000 --normalize
# python src/train_predictors.py --env "drone" --predictor FE_NeuralODE --data_type "random"  --seed 0 --steps 5000 --normalize
# python src/train_predictors.py --env "drone" --predictor FE_Residuals --data_type "random"  --seed 0 --steps 5000 --normalize
# python src/train_predictors.py --env "drone" --predictor FE_NeuralODE_Residuals --data_type "random"  --seed 0 --steps 5000 --normalize

#python src/evaluate_predictors.py --env "drone" --data_type "random" --normalize
#python src/evaluate_drone_transfer.py --normalize

#
#
#
export PYTHONPATH=$PATHONPATH:`pwd`
nohup python src/MPC_Drone.py --alg NeuralODE > output1.log 2>&1 &
nohup python src/MPC_Drone.py --alg FE_NeuralODE > output2.log 2>&1 &
nohup python src/MPC_Drone.py --alg FE_NeuralODE_Residuals > output3.log 2>&1 &