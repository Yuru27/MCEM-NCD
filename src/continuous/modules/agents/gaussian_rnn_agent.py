import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Normal, Independent


class GaussianRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GaussianRNNAgent, self).__init__()
        self.args = args

        self.num_actions = args.n_actions
        self.clip_stddev = args.clip_stddev > 0
        self.clip_std_threshold = np.log(args.clip_stddev)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.mean = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.log_std = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # Action rescaling
        self.action_max = torch.tensor(
            self.args.action_spaces[0].high, device=self.args.device)
        self.action_min = torch.tensor(
            self.args.action_spaces[0].low, device=self.args.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.fc1(inputs))
        x = x.reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        mean = torch.tanh(self.mean(h))
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + \
            self.action_min
        log_std = self.log_std(h)
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
        return mean, log_std, h

    def sample(self, states, hidden_state, num_samples=1):

        mean, log_std, hidden_states = self.forward(states, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = normal.sample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)

        log_prob = normal.log_prob(action)
        entropy = normal.entropy()
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)
            entropy.unsqueeze(-1)

        return action, log_prob, entropy, hidden_states

    def test_actions(self, states, hidden_state):

        mean, log_std, hidden_states = self.forward(states, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = mean.unsqueeze(0).detach()
        action = torch.clamp(action, self.action_min, self.action_max)

        log_prob = normal.log_prob(action)
        entropy = normal.entropy()
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)
            entropy.unsqueeze(-1)

        return action, log_prob, entropy, hidden_states

    def rsample(self, states, hidden_state, num_samples=1):

        mean, log_std, hidden_states = self.forward(states, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        action = normal.rsample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)
        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        return action, log_prob, hidden_states

    def log_prob(self, states, hidden_state, actions, num_samples=1):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std, hidden_states = self.forward(states, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        log_prob = normal.log_prob(actions)

        return log_prob, hidden_states

    def clamp_actions(self, chosen_actions):
        for _aid in range(self.args.n_agents):
            for _actid in range(self.args.action_spaces[_aid].shape[0]):
                chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
                                                       np.asscalar(self.args.action_spaces[_aid].high[_actid]))
