from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)
        
        # Parameters
        n_ranges = 8   # Number of lambda ranges
        n_metrics = 6  # Number of metrics

        self.vel_data = np.zeros((num_env,1000))
        self.vel_mean = np.zeros(num_env)
        self.vel_max = np.zeros(num_env)
        self.lambda_r = np.zeros(num_env)
        self.step = np.zeros(num_env, dtype=int)
        self.episode = np.zeros(num_env, dtype=int)
        # Lambda ranges
        self.cp_rng = cp_rng    # External variable
        self.n_lambda_rng = n_ranges
        self.lambda_ranges = np.zeros((self.n_lambda_rng, n_metrics))
        for i in range(self.n_lambda_rng):
            self.lambda_ranges[i,0] = self.cp_rng[0] + (i)*(self.cp_rng[1] - self.cp_rng[0]) / self.n_lambda_rng
            self.lambda_ranges[i,1] = self.cp_rng[0] + (i+1)*(self.cp_rng[1] - self.cp_rng[0]) / self.n_lambda_rng
#         print(self.lambda_ranges)
        # Success ranges
        self.success_ranges = np.zeros((self.n_lambda_rng,100))
        self.episode_ranges = np.zeros(self.n_lambda_rng, dtype=int)

    # Lambda ranges
    def _on_step(self) -> bool:
        # For all environments
        for e in range(self.model.env.buf_dones.shape[0]):
            # Velocity norm at every step
            self.vel_data[e,self.step[e]] = np.sqrt(self.model.env.buf_obs[None][e,2]*self.model.env.buf_obs[None][e,2] + self.model.env.buf_obs[None][e,3]*self.model.env.buf_obs[None][e,3])
            self.step[e] += 1

            if self.model.env.buf_dones[e]:
                self.vel_mean[e] = np.sum(self.vel_data[e])/self.step[e]
                self.lambda_r[e] = self.model.env.buf_obs[None][e,8]
                self.vel_max[e] = np.max(self.vel_data[e])
                # If successful landing
#                 if self.model.env.buf_rews[e] > 90:
#                     self.success_data[e, self.episode[e]%100] = 1
#                 else:
#                     self.success_data[e, self.episode[e]%100] = 0

                # Lambda ranges START
                for i in range(self.n_lambda_rng):
                    if (self.lambda_r[e] >= self.lambda_ranges[i,0]) and (self.lambda_r[e] <= self.lambda_ranges[i,1]):
                        self.lambda_ranges[i,2] = self.vel_mean[e]
                        self.lambda_ranges[i,4] = self.lambda_r[e]
                        self.lambda_ranges[i,5] = self.vel_max[e]
                        # If successful landing
                        if self.model.env.buf_rews[e] > 90:
                            self.success_ranges[i, self.episode_ranges[i]%100] = 1
                        else:
                            self.success_ranges[i, self.episode_ranges[i]%100] = 0
                        self.lambda_ranges[i,3] = np.mean(self.success_ranges[i])

                        self.episode_ranges[i] += 1
                # Lambda ranges END

                # Add an episode to the environment
                self.episode[e] += 1
                # Reset vel buffer
                self.vel_data[e] = np.zeros(1000)
                self.step[e] = 0

        for i in range(self.n_lambda_rng):
            tag = 'Mean Velocity. Lambda: '+ str(self.lambda_ranges[i,0]) + " : " + str(self.lambda_ranges[i,1])
            value = self.lambda_ranges[i,2]
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            
            tag = 'Max Velocity. Lambda: '+ str(self.lambda_ranges[i,0]) + " : " + str(self.lambda_ranges[i,1])
            value = self.lambda_ranges[i,5]
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)

            tag = 'Success. Lambda: '+ str(self.lambda_ranges[i,0]) + " : " + str(self.lambda_ranges[i,1])
            value = self.lambda_ranges[i,3]
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)

            tag = 'Lambda. Lambda: '+ str(self.lambda_ranges[i,0]) + " : " + str(self.lambda_ranges[i,1])
            value = self.lambda_ranges[i,4]
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)

        return True
