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
    # return -F.nll_loss(F.softmax(policy_logits), actions, reduction='none')
    # return -torch.nn.NLLLoss(torch.nn.LogSoftmax(policy_logits, dim=-1), actions)
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

    # with torch.no_grad():
    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.min(clip_rho_threshold, rhos)
    else:
        clipped_rhos = rhos

    cs = torch.min(torch.ones(rhos.shape), rhos)
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat((values[1:], bootstrap_value.unsqueeze(0)), dim=0)
    # values_t_plus_1 = torch.cat((values, bootstrap_value), dim=0)
    #deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # Note that all sequences are reversed, computation starts from the back.
    # V-trace vs are calculated through a scan from the back to the beginning
    # of the given trajectory.
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # sequences = (discounts, cs, deltas)

    initial_values = torch.zeros_like(bootstrap_value)
    acc = initial_values
    acc.requires_grad_(False)
    seq_len = discounts.shape[0]
    for i in range(seq_len - 1, -1, -1):
        print(i)
        discount_t, c_t, delta_t = discounts[i], cs[i], deltas[i]
        acc = delta_t + discount_t * c_t * acc

    vs_minus_v_xs = acc
    # vs_minus_v_xs = []
    # for i in range(seq_len):
    #     # v_s = values[i].clone()  # Add V(x_s) to get v_s.
    #     vs_t = 0
    #     for j in range(i, seq_len):
    #         basic_td = rewards[j] + discounts[j] * values_t_plus_1[j] - values[j]
    #         vs_t += torch.prod(discounts[i:j], dim=0) * torch.prod(cs[i:j], dim=0) * clipped_rhos[j] * basic_td
    #     vs_minus_v_xs.append(vs_t)
    #
    #
    # vs_minus_v_xs = torch.stack(vs_minus_v_xs, dim=0)
    # print(vs_minus_v_xs)
    # vs = vs_minus_v_xs
    vs = torch.add(vs_minus_v_xs, values)
    # print(vs)
    # Advantage for policy gradient.
    vs_t_plus_1 = torch.cat((vs[1:], bootstrap_value.unsqueeze(0)), dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.min(clip_pg_rho_threshold, rhos)
    else:
        clipped_pg_rhos = rhos

    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
    # print('pg_advantages', pg_advantages)
    # Make sure no gradients backpropagated through the returned values.
    # print(pg_advantages)
    return vs.detach(), pg_advantages.detach()