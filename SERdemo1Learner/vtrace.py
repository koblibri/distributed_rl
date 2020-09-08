from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


def log_probs_from_logits_and_actions(policy_logits, actions):
    # policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    # actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    num_actions = 12
    assert len(policy_logits.shape) == 3
    assert len(actions.shape) == 2
    # a = torch.log_softmax(policy_logits, 2)
    # print('policy_logits', policy_logits.shape)
    # print('actions', actions.shape)

    # one_hot = torch.zeros(len(actions), num_actions).scatter_(1, actions, 1)
    # one_hot = F.one_hot(actions, num_classes=num_actions)
    # neg_log_prob = torch.sum(-torch.log(F.softmax(policy_logits, dim=1)) * one_hot, dim=1)
    return -F.cross_entropy(policy_logits, actions, reduction='none')
    # a = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=policy_logits, labels=actions)
    # print(a)
    # return -neg_log_prob

    # return -neg_log_prob


def from_logits(behaviour_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, bootstrap_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):

    # Make sure tensor ranks are as expected.
    # The rest will be checked by from_action_log_probs.
    assert len(behaviour_policy_logits.shape) == 3
    # print('a', actions.shape)
    assert len(target_policy_logits.shape) == 3
    assert len(actions.shape) == 2

    target_action_log_probs = log_probs_from_logits_and_actions(
        target_policy_logits, actions)
    behaviour_action_log_probs = log_probs_from_logits_and_actions(
        behaviour_policy_logits, actions)
    log_rhos = target_action_log_probs - behaviour_action_log_probs
    vs, pg_advantages = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)
    return vs, pg_advantages


def from_importance_weights(
        log_rhos, discounts, rewards, values, bootstrap_value,
        clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):


    if clip_rho_threshold is not None:
        clip_rho_threshold = torch.tensor(clip_rho_threshold, dtype=torch.float32, device=log_rhos.device)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold, dtype=torch.float32, device=log_rhos.device)

    # Make sure tensor ranks are consistent.
    rho_rank = len(log_rhos.shape)  # Usually 2.
    assert len(values.shape) == rho_rank
    assert len(bootstrap_value.shape) == rho_rank - 1
    assert len(discounts.shape) == rho_rank
    assert len(rewards.shape) == rho_rank

    if clip_rho_threshold is not None:
        assert len(clip_rho_threshold.shape) == 0
    if clip_pg_rho_threshold is not None:
        assert len(clip_pg_rho_threshold.shape) == 0

    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.min(clip_rho_threshold, rhos)
        else:
            clipped_rhos = rhos

        cs = torch.min(torch.ones_like(rhos), rhos)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat((values, bootstrap_value.unsqueeze(0)), dim=0)
        #deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        # Note that all sequences are reversed, computation starts from the back.
        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        seq_len = discounts.shape[0]
        vs = []
        for i in range(seq_len):
            v_s = values[i].clone()
            for j in range(i, seq_len):
                v_s += (torch.prod(discounts[i:j], dim=0) * torch.prod(cs[i:j], dim=0) * clipped_rhos[j] *
                        (rewards[j] + discounts[j] * values_t_plus_1[j + 1] - values[j]))
            vs.append(v_s)
        vs = torch.stack(vs, dim=0)
        # Advantage for policy gradient.
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.min(clip_pg_rho_threshold, rhos)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = (
                clipped_pg_rhos * (rewards + discounts * torch.cat(
            (vs[1:], bootstrap_value.unsqueeze(0)), dim=0) - values))

        # Make sure no gradients backpropagated through the returned values.
        return vs, pg_advantages