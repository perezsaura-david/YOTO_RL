import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom feature extractor for YOTO
class YotoFE(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, lambda_dim: int = 1):
        super(YotoFE, self).__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        # First
        self.fc1 = nn.Sequential(nn.Linear(obs_dim, 512), 
                                 nn.ReLU()
        )
        # Mean mlp
        self.mean_mlp = nn.Sequential(nn.Linear(lambda_dim, 128), 
                                      nn.ReLU(),
                                      nn.Linear(128, 512),
                                      nn.ReLU()
        )
        # Std mlp
        self.std_mlp = nn.Sequential(nn.Linear(lambda_dim, 128), 
                                      nn.ReLU(),
                                      nn.Linear(128, 512),
                                      nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs_dim = observations.shape[1]
        envobs = observations[:,:obs_dim]
        # print('env obs:', envobs.shape) # torch.Size([1, 4])
        lambda_p = observations[:,-1:]
        # print('lambda:', lambda_p.shape) # torch.Size([1, 1])

        out1     = self.fc1(observations)
        out_std  = self.std_mlp(lambda_p)
        out_mean = self.mean_mlp(lambda_p)
        out2     = th.multiply(out_std, out1)
        # print(out2.shape) # torch.Size([1, 512])
        extracted_features = th.add(out_mean, out2)
        # print('extracted features:', extracted_features.shape) # torch.Size([1, 512])

        return extracted_features

# Custom environment wrapper for YOTO
class YotoEnv(gym.Wrapper):
    def __init__(self, env, lambda_0, train, lambda_eps, lambda_rng):
        super(YotoEnv, self).__init__(env)
        self.lambda_eps = lambda_eps
        self.l_eps = 0
        self.train = train
        self.lambda_rng = lambda_rng

        if train:
            self.new_lambda()
        else:
            self.lambda_r = lambda_0
        n_obs = self.observation_space.shape[0] + 1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        
        self.prev_obs = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

#         reward = reduce_steps(self.lambda_r, reward)
        reward = max_velocity(self.lambda_r, reward, obs)
#         reward = landing_angle(self.lambda_r, reward, obs)
#         reward = vel_shaping(self.lambda, reward, obs, self.prev_obs)
#         self.prev_obs = obs
#         reward = control_velocity(self.lambda_r, reward, obs) 
        # print('YOTO rwd: ', reward)

        lambda_r = np.array([self.lambda_r])
        obs = np.append(obs, lambda_r, axis=0)

        # print(lambda_r)

        if self.train:
            if done:
                self.l_eps += 1
            if self.l_eps == self.lambda_eps:
                self.new_lambda()
                self.l_eps = 0
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        lambda_r = np.array([self.lambda_r])
        obs = np.append(obs, lambda_r, axis=0)
        
        self.prev_obs = None
        return obs

    def new_lambda(self):
        self.lambda_r = sample_log_uniform(low=self.lambda_rng[0], high=self.lambda_rng[1], size=1)[0]
        # print('New lambda: ', self.lambda_r)

def sample_log_uniform(low=0.01, high=2.0, size=1):
#     return np.exp(np.random.uniform(np.log(low), np.log(high), size))
    # return np.random.uniform(low, high, size)
    if low <= 0:
        return np.random.uniform(low, high, size)
    else:
        return np.exp(np.random.uniform(np.log(low), np.log(high), size))
