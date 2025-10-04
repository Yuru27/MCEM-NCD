import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.ncd_critic import NCDCritic
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from utils.rl_utils import compute_q_retraces
from torch.distributions import Categorical
from components.transforms import OneHot

class MCEMNCDLearner:
    def __init__(self, Mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = Mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(self.mac.parameters())
        self.agent_params += list(self.target_mac.parameters())

        self.critic = NCDCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.num_sampling = args.num_sampling
        self.topk = args.topk

        self.mixer = None
        if (
            args.mixer is not None and self.args.n_agents > 1
        ): 
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(
                params=self.agent_params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(
                params=self.agent_params,
                lr=args.lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            raise Exception(
                "unknown optimizer {}".format(
                    getattr(self.args, "optimizer", "rmsprop")
                )
            )

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(
                params=self.critic_params,
                lr=args.critic_lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(
                params=self.critic_params,
                lr=args.critic_lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            raise Exception(
                "unknown optimizer {}".format(
                    getattr(self.args, "optimizer", "rmsprop")
                )
            )

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, off=False):
        # Get the relevant data
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"]
        behaviour = batch["probs"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        states = batch["state"]

        target = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(max_t):
            agent_outs = self.mac.forward(batch, t=t)
            target.append(agent_outs)
        target = th.stack(target, dim=1)

        with th.no_grad():
            ratio = target/behaviour
            sample_actions = Categorical(target).sample().unsqueeze(-1)

            # build q
            inputs = self.target_critic._build_inputs(batch, bs, max_t)
            q1 = self.target_critic(inputs).detach()

            q_vals = th.gather(q1, 3, index=actions)
            target_q_vals = self.target_mixer(q_vals, states)

            s_vals = th.gather(q1, 3, index=sample_actions)
            target_s_vals = self.target_mixer(s_vals, states)

 
            q_retraces = compute_q_retraces(
                target_q_vals, target_s_vals, rewards, actions, mask, ratio, self.args.gamma, self.args.tb_lambda, self.args.l_c
            )


        # Train the critic
        # Current Q network forward
        inputs = self.critic._build_inputs(batch, bs, max_t)
        q1 = self.critic(inputs)
        q_vals = th.gather(q1, 3, index=actions)
        q_taken = self.mixer(q_vals, states)

        critic_loss = ((q_taken[:, :-1] - q_retraces[:, :-1]) * mask).pow(
            2
        ).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        # Train the actor
        if off:
            mac_logist = Categorical(probs=target).logits
            target = []
            onehot = OneHot(out_dim=self.args.n_actions)
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.target_mac.forward(batch, t=t)
                target.append(agent_outs)

            target = th.stack(target, dim=1)


            g = Categorical(target)
            m_logits = g.logits
            m_entropy = g.entropy().mean(dim=-1, keepdim=True)

            sample_actions = [
                g.sample() for x in range(self.num_sampling)
            ]
            # actions_onehot = onehot.transform(
            #     th.stack(sample_actions).transpose(
            #         1, 0).transpose(2, 1).unsqueeze(-1)
            # )

            actions_onehot = onehot.transform(th.stack(sample_actions).permute(1,2,0,3).unsqueeze(-1))

            # sample_actions = (
            #     th.stack(sample_actions).unsqueeze(-1).transpose(1,
            #                                                      0).transpose(2, 1)
            # )
            sample_actions = (th.stack(sample_actions).permute(1,2,0,3).unsqueeze(-1))
 
            # states = (
            #     batch["state"]
            #     .squeeze()
            #     .expand((self.num_sampling), -1, -1, -1)
            #     .transpose(1, 0)
            #     .transpose(2, 1)
            # )

            states = ( 
                batch["state"].unsqueeze(2).repeat_interleave(repeats=self.num_sampling,dim=2)
            )

            # inputs = (
            #     self.critic._build_inputs(batch, bs, max_t)
            #     .squeeze()
            #     .expand((self.num_sampling), -1, -1, -1, -1)
            #     .transpose(1, 0)
            #     .transpose(2, 1)
            # )

            inputs = (
                self.critic._build_inputs(batch, bs, max_t).unsqueeze(2).repeat_interleave(repeats=self.num_sampling,dim=2)

            )

            q1 = []
            for t in range(batch.max_seq_length):
                q1.append(self.critic(inputs[:, t]))

            q1 = th.stack(q1, dim=1).detach()
            q_vals = th.gather(q1, 4, index=sample_actions)
            q_taken = self.mixer(q_vals, states).detach()

            q_taken = q_taken.reshape(
                batch.batch_size,
                (batch.max_seq_length),
                (self.num_sampling),
                -1,
            ) 

            m_onehot = actions_onehot[
                th.arange(q_taken.shape[0])[:, None, None],
                th.arange(q_taken.shape[1])[None, :, None],
                q_taken.squeeze(-1).topk(k=self.topk, dim=-1)[1]
            ]

            proposal_loss = -((m_onehot[:, :-1] * m_logits[:, :-1].unsqueeze(2)).sum(dim=-1)
                * mask.unsqueeze(2)).sum() / mask.sum()/self.topk

            entropy_mask = copy.deepcopy(mask)
            entropy_loss = (m_entropy[:, :-1] * entropy_mask).sum() / entropy_mask.sum()
            proposal_loss = proposal_loss - self.args.entropy_coef * entropy_loss / entropy_loss.item()

            actor_loss = -((m_onehot[:, :-1] * mac_logist[:, :-1].unsqueeze(2)).sum(dim=-1)
                * mask.unsqueeze(2)).sum() / mask.sum()/self.topk

            loss = proposal_loss + actor_loss

            self.agent_optimiser.zero_grad()
            loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        # target_update
        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # log
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm",
                                 critic_grad_norm.item(), t_env)
            self.logger.log_stat(
                "target_vals",
                (q_retraces[:, :-1] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("num_sampling", self.num_sampling, t_env)
            
            if off:
                self.logger.log_stat("loss", loss.item(), t_env)
                self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
                self.logger.log_stat(
                    "agent_grad_norm", agent_grad_norm.item(), t_env)
                agent_mask = mask.repeat(1, 1, self.n_agents)
                self.logger.log_stat(
                    "pi_max",
                    (th.exp(m_logits[:, :-1]).max(dim=-1)[0] * agent_mask).sum().item()
                    / agent_mask.sum().item(),
                    t_env,
                )
                self.log_stats_t = t_env

    def build_exp_v(self, target_q_vals, mac_out, states, dim=3):
        target_exp_v_vals = th.sum(target_q_vals * mac_out, dim)
        target_exp_v_vals = self.target_mixer.forward(
            target_exp_v_vals, states)
        return target_exp_v_vals

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _build_obs(self, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        ts = batch.max_seq_length
        inputs = []
        inputs.append(batch["obs"])  # b1av
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, ts, -1, -1)
            )
        inputs = th.cat([x.reshape(bs, ts, self.n_agents, -1)
                        for x in inputs], dim=-1)
        return inputs

    def cuda(self, device="cuda"):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.to(device=device)
        self.target_critic.to(device=device)
        if self.mixer is not None:
            self.mixer.to(device=device)
            self.target_mixer.to(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage)
        )
