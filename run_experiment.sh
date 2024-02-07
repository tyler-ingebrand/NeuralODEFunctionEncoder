#!/bin/bash


#python src/vanderpol.py --train_type "uniform" --test_type "uniform" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "uniform" --test_type "trajectory" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "uniform" --test_type "trajectory_ls" --use_approximate_model True --grad_steps 1000 --plot_time 500

#python src/vanderpol.py --train_type "trajectory" --test_type "trajectory" --use_approximate_model True --grad_steps 1000 --plot_time 500
#python src/vanderpol.py --train_type "trajectory" --test_type "trajectory_ls" --use_approximate_model True --grad_steps 1000 --plot_time 500

python src/train_policy.py --env "Ant-v3" --num-timesteps 10000000 --alg sac
python src/train_policy.py --env "HalfCheetah-v3" --num-timesteps 10000000 --alg sac

python src/gather_trajectories.py --alg sac --env "Ant-v3" --num-timesteps 10000000 # note you can run these in parallel if you want
python src/gather_trajectories.py --alg sac --env "HalfCheetah-v3" --num-timesteps 10000000