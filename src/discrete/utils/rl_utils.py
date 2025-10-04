import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def compute_q_retraces(target_q_vals, v_vals, rewards, actions, mask, ratio, gamma, tb_lambda, l_clamp):
    """
    Compute Q retraces for the given target Q values, value estimates, rewards, actions, mask, ratio, and discount factor.

    Args:
        target_q_vals (Tensor): Target Q values.
        v_vals (Tensor): Value estimates.
        rewards (Tensor): Rewards.
        actions (Tensor): Actions taken.
        mask (Tensor): Mask indicating valid steps.
        ratio (Tensor): Importance sampling ratios.
        gamma (float): Discount factor.

    Returns:
        Tensor: Computed Q retraces.
    """
    time_steps = target_q_vals.size(1) - 1
    q_retraces = th.zeros_like(v_vals)
    tmp_retraces = v_vals[:, -1]
    q_retraces[:, -1] = v_vals[:, -1]
    ratio_gather = ratio.gather(-1, actions).squeeze()

    for idx in reversed(range(time_steps)):
        q_retraces[:, idx] = rewards[:, idx] + gamma * mask[:, idx] * tmp_retraces
        tmp_retraces = (
            tb_lambda
            * ratio_gather[:, idx].prod(dim=-1, keepdim=True).clamp(min=l_clamp,max=1)
            * (q_retraces[:, idx] - target_q_vals[:, idx])
            + v_vals[:, idx]
        )

    return q_retraces