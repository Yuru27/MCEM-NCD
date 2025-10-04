import torch as th
import numpy as np
from .basic_controller import BasicMAC


class MCEMNCDMAC(BasicMAC):

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, past_actions=None, critic=None,
                       target_mac=False, explore_agent_ids=None):
        avail_actions = ep_batch["avail_actions"][bs, t_ep]

        if t_ep is not None and t_ep > 0:
            past_actions = ep_batch["actions"][:, t_ep-1]

        # Note batch_size_run is set to be 1 in our experiments
        if self.args.agent in ["grnn"]:
            if not test_mode:
                chosen_actions, log_prob, _ = self.sample(ep_batch[bs],
                                                          t_ep,
                                                          hidden_states=self.hidden_states[bs],
                                                          select_actions=True)
            else:
                chosen_actions, log_prob, _ = self.test_actions(ep_batch[bs],
                                                                t_ep,
                                                                hidden_states=self.hidden_states[bs],
                                                                select_actions=True)
            pass
        else:
            raise Exception(
                "No known agent type selected for cqmix! ({})".format(self.args.agent))
        return chosen_actions, log_prob

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None, hidden_states=None, select_actions=False, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        mean, log_std, hidden_states = self.agent(
            agent_inputs, self.hidden_states, actions=actions)
        if select_actions:
            self.hidden_states = hidden_states
            return mean, log_std

    def sample(self, ep_batch, t, hidden_states=None, select_actions=False, num_samples=1):

        agent_inputs = self._build_inputs(ep_batch, t)

        action, log_prob, entropy, hidden_states = self.agent.sample(
            agent_inputs, self.hidden_states,  num_samples)
        if select_actions:
            self.hidden_states = hidden_states
            return action, log_prob, entropy

    def test_actions(self, ep_batch, t, hidden_states=None, select_actions=False):

        agent_inputs = self._build_inputs(ep_batch, t)

        action, log_prob, entropy, hidden_states = self.agent.test_actions(
            agent_inputs, self.hidden_states)
        if select_actions:
            self.hidden_states = hidden_states
            return action, log_prob, entropy

    def log_prob(self, ep_batch, t, actions=None, hidden_states=None, select_actions=False, test_mode=False, num_samples=1):

        agent_inputs = self._build_inputs(ep_batch, t)

        log_prob, hidden_states = self.agent.log_prob(
            agent_inputs, self.hidden_states, actions, num_samples)
        if select_actions:
            self.hidden_states = hidden_states
            return log_prob


    def _build_inputs(self, batch, t, target_mac=False, last_target_action=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(
                0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1)
                        for x in inputs], dim=1)

        return inputs.view(bs, self.n_agents, -1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            if getattr(self.args, "discretize_actions", False):
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def clamp_actions(self, chosen_actions):
        for _aid in range(self.n_agents):
            for _actid in range(self.args.action_spaces[_aid].shape[0]):
                chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
                                                       np.asscalar(self.args.action_spaces[_aid].high[_actid]))

        return chosen_actions
