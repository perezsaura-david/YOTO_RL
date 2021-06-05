import torch as th
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import time
from datetime import datetime

# Training parameters
num_env = 1
max_steps = int(1e6)
log_int = 10
cp_rng = [0.0,2.0]
lambda_eps = 1

wrapper_params = {
    'lambda_0': 0,
    'train': True,
    'lambda_eps': lambda_eps,
    'lambda_rng': cp_rng
    }

# Model file name
cprng = str(cp_rng[0])+':'+str(cp_rng[1])

folder_name = "models/"
# env_name    = "ppo2_lunarlander/yoto/wrapper/"
# rfun_name   = "velshap/"
model_name  = "yoto_"+str(cprng)+"_eps_"+str(max_steps)
# model_file  = folder_name+env_name+rfun_name+model_name
model_file  = folder_name + model_name
tb_log      = "TB/"

print(model_file)

# Environment
yoto_env = make_vec_env_mod('LunarLander-v2', n_envs=num_env, wrapper_class=YotoEnv, wrapper_kwargs=wrapper_params)  
# Custom actor (pi) and value function (vf) networks of two layers of size 32 each with Relu activation function
policy_kwargs = dict(features_extractor_class=YotoFE,
                     features_extractor_kwargs=dict(features_dim=512),
                     activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128, 128], 
                                    vf=[32, 32])])     
# Model       
model = PPO("MlpPolicy", yoto_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tb_log)
# model = PPO("MlpPolicy", yoto_env, policy_kwargs=policy_kwargs, verbose=1)
print(model_file)
# Training
now = datetime.now()
print('Start training:', now.time())
time_0 = time.time()
%tensorboard --logdir TB
model.learn(total_timesteps=max_steps, log_interval=log_int, callback=TensorboardCallback())
# model.learn(total_timesteps=max_steps, log_interval=log_int)
time_training = time.time() - time_0
print('Training time:', time_training/60, 'min')
model.save(model_file)
# del model # remove to demonstrate saving and loading
print(model_file)
