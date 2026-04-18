

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda

# from learn_framework import LFramework
# import rl.graph_search.beam_search as search
# import utils.ops as ops
# from utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn, ra):
        super(PolicyGradient, self).__init__(args, kg, pn, ra)

        # Training hyperparameters
        self.relation_only = args.relation_only
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval
        self.supconloss = SupConLoss(temperature=0.07, contrast_mode="all", base_temperature=0.07).to(torch.device("cuda"))

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

    def reward_fun(self, e1, r, e2, pred_e2):
        return (pred_e2 == e2).float()

    def loss(self, mini_batch):

        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        advantages = output['advantages']
        critic_loss = output['critic_loss']
        critic_loss = torch.stack(critic_loss)
        sim_rates = output['sim_rate']
        hit_rates = output['hit_rate']
        rel_reward = output['rel_reward']
        rel_sim = output['entity_sim']
        path_trace = output['path_trace']
        ra_value_list = output['ra_value_list']
        cl_x = output['cl_x']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        done = (pred_e2 == e2).float()

        # tail_emb = self.kg.get_entity_embeddings(pred_e2)
        # tail_emb = F.normalize(tail_emb, dim=1)
        # # cl_x = torch.cat((self.kg.get_entity_embeddings(e1), self.kg.get_relation_embeddings(r)), -1)
        # cl_x = F.normalize(cl_x, dim=1)
        # features1 = torch.cat((cl_x.unsqueeze(1), tail_emb.unsqueeze(1)), dim=1)
        # supconloss1 = self.supconloss(features1, labels=e2, mask=None)

        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward

        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R # (R + advantages[i]) / 2

        # potential和origin间别差太大
        potential_discount = 0.5
        origin_discount = 0.5
        for i in range(self.num_rollout_steps - 1, -1, -1):
            relative_potential = torch.relu(rel_reward[i])
            Rs = origin_discount * cum_discounted_rewards[i]
            Rp = potential_discount * relative_potential
            # cum_discounted_rewards[i] = done + (1 - done) * (Rs + Rp)
            # cum_discounted_rewards[i] = (rel_sim + relative_potential + cum_discounted_rewards[i] * (1 - done)) / 3 + done
            # cum_discounted_rewards[i] = (rel_sim + 1) * relative_potential + cum_discounted_rewards[i]
            # 早期需要探索，因此早期的探索占比高点
            cum_discounted_rewards[i] = (Rs + Rp)
            # cum_discounted_rewards[i] = advantages[i]
            # torch.stack([cum_discounted_rewards[-1], final_reward], 0).cpu().data.numpy()

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean() # + supconloss1
        pt_loss = (pt_loss - entropy * self.beta).mean() # + supconloss1

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        loss_dict['critic_loss'] = float(critic_loss.mean())
        loss_dict['sim_rates'] = float(sim_rates)
        loss_dict['hit_rates'] = float(hit_rates)
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def compute_advantage(self, rewards, values, dones, rel_sim, gamma=1.0, lam=0.095):
        # rewards = torch.stack(rewards, 0)
        # values = torch.stack(values, 0)
        # dones = torch.stack(dones, 0)
        # rel_sim = torch.stack(rel_sim, 0)
        # # rewards = (rel_sim + rewards) / 2
        # next_value = torch.tensor(0.0)
        # gae = torch.tensor(0.0)
        # advantages = torch.zeros_like(rewards)
        # for t in reversed(range(len(rewards))):
        #     delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        #     gae = delta + gamma * lam * (1 - dones[t]) * gae
        #     advantages[t] = gae
        #     next_value = values[t]
        #
        rewards = torch.stack(rewards, 0)
        values = torch.stack(values, 0)
        dones = torch.stack(dones, 0)
        rel_sim = torch.stack(rel_sim, 0)
        advantages = torch.zeros_like(rewards)
        # values = (torch.relu(rel_sim) + values) / 2
        # rewards = (rel_sim + rewards) / 2
        # rewards = torch.relu(rel_sim) + rewards
        next_value = torch.tensor(0.0)
        # next_value = values[-1]
        # gae = torch.tensor(0.0)
        # for t in reversed(range(len(values))):
        for t in reversed(range(len(values) - 1)):
        #     delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            delta = (rewards[t] + gamma * next_value) / 2 - values[t]
            # delta = rewards[t] - values[t]
            # delta = next_value - values[t]
            # gae = delta + gamma * lam * (1 - dones[t]) * gae
            # R = self.gamma * R + rewards[t]
            # delta = R / (len(values) - 1 - t) - values[t]
            advantages[t] = delta
            next_value = values[t]

        potentials = torch.zeros_like(rewards)
        # advantages = torch.sigmoid(advantages)
        # rel_sim = torch.sigmoid(rel_sim)
        # next_rel_sim = torch.tensor(0.0)
        next_rel_sim = rel_sim[-1]
        for t in reversed(range(len(rel_sim) - 1)):
            # delta = rewards[t] + (gamma * next_rel_sim - rel_sim[t]) * (1 - dones[t])
            delta = next_rel_sim - rel_sim[t]
            potentials[t] = delta
            next_rel_sim = rel_sim[t]

        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = rewards[-1]
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i] + potentials[i]
            cum_discounted_rewards[i] = (R + torch.clamp(advantages[i], min=0.0)) / 2
            # cum_discounted_rewards[i] = torch.clamp(R, min=0)

        return cum_discounted_rewards, torch.clamp(advantages, min=0)

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn, ra = self.kg, self.mdl, self.ra

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        inv_q = kg.get_inv_relation_id(q)
        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)
        # ra.actor.initialize_path((r_s, e_s), kg)

        MA_entity, RA_entity, MA_value_list, MA_reward_list = [], [], [], []
        MA_dones = []
        last_e = e_s

        # rel_sim = self.fn_kg.get_relation_embeddings(path_trace[-1][0])
        rel_sim = torch.tensor(0.0)
        # rel_sim = torch.zeros_like(self.fn_kg.get_relation_embeddings(path_trace[-1][0]))
        rel_sim_list = []

        ra_state, ra_next_state, ra_learn_reward, ra_done, ra_log_prob, critic_loss_list = [], [], [], [], [], []

        # MA_value_list.append(torch.sigmoid(ra.fact_score(e_s, q, e_s, e_s, kg).detach()))
        # MA_value_list.append(torch.tanh(ra.fact_score(e_s, q, e_s, e_s, kg).detach()))
        # MA_value_list.append(ra.fact_score(e_s, q, e_s, e_s, kg).detach())
        # init_MA_value = ra.fact_score(e_s, q, e_s, e_s, kg).detach()
        # init_MA_value = torch.zeros_like(e_s)
        # MA_value_list.append(init_MA_value)

        # rel_sim_score = torch.cosine_similarity(rel_sim, self.fn_kg.get_relation_embeddings(q)).detach()
        # rel_sim_list.append(rel_sim_score)
        # init_rel_sim_score = torch.cosine_similarity(rel_sim, self.fn_kg.get_relation_embeddings(q)).detach()
        init_rel_sim_score = torch.zeros_like(q)
        rel_sim_list.append(init_rel_sim_score)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]

            # main agent
            db_outcomes, inv_offset, policy_entropy, bucket, H = pn.transit(
                e, obs, ra, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            # e, obs, kg, ra_next_e, bucket, ra, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            # # RA sample
            # # ra_action, next_e, bucket, space_prob, space_entropy = ra.perceive(e, obs, kg)
            # # ra_db_outcomes, bucket, ra_entropy = ra.perceive(e, obs, kg)
            # ra_db_outcomes, ra_entropy, ra_inv_offset = ra.perceive(e, action[1], obs, kg, bucket)
            # ra_sample_outcome = self.sample_action(ra_db_outcomes, ra_inv_offset)
            # ra_action = ra_sample_outcome['action_sample']
            # # ra.actor.update_path(ra_action, kg)
            # ra_action_prob = ra_sample_outcome['action_prob']
            # ra_next_e = ra_action[1]
            ra_action_prob, ra_next_e, ra_db_outcomes, ra_entropy = ra.perceive(e, H, action[1], obs, kg, bucket)

            # ra_reward = self.reward_fun(e_s, q, e_t, e)
            # ra_reward = self.reward_fun(e_s, q, e_t, e_pred)      #
            # ra_reward = self.reward_fun(e, q, e_t, pred_action)          #
            # ra_reward = self.fn.forward_fact(e_s, q, pred_action, self.fn_kg).squeeze(1)
            # ra_reward = self.reward_fun(e, q, e_t, action[1])     #
            MA_reward = self.reward_fun(e_s, q, e_t, action[1])   #
            RA_reward = self.reward_fun(e_s, q, e_t, ra_next_e)        # 1
            # ra_reward = (next_e == e_t).float()
            # ra_reward = self.reward_fun(e_s, q, e_t, pred_action)
            # ra_reward = torch.cosine_similarity(kg.get_entity_embeddings(pred_action), kg.get_entity_embeddings(e_t)).abs().detach()
            RA_entity.append(ra_next_e)    # 2
            MA_entity.append(action[1])

            RA_done = (ra_next_e == e_t).float()  # 3
            MA_done = (action[1] == e_t).float()
            # if t == num_steps - 1:
            #     MA_done = torch.ones_like(MA_done)
            #     RA_done = torch.ones_like(RA_done)

            # state = [e_s, q, e, e_t, action]
            # state = [e_s, q, e, e_t, last_e]
            MA_state = [MA_reward, action, MA_done]
            RA_state = [e_s, q, e, e_t, last_e, RA_reward, ra_next_e, RA_done]

            # state = [kg.get_entity_embeddings(e_s), kg.get_relation_embeddings(q), kg.get_entity_embeddings(e)]
            # next_state = [kg.get_entity_embeddings(e_s), kg.get_relation_embeddings(q), kg.get_entity_embeddings(action[1])]

            # ra_state.append(state)
            # ra_next_state.append(next_e)
            # ra_learn_reward.append(ra_reward)
            # ra_done.append(done)
            # ra_log_prob.append(ops.safe_log(space_prob))
            # with torch.no_grad():
                # MA_value_list.append(torch.sigmoid(ra.fact_score(e_s, q, e, action[1], kg).detach()))
            # MA_value_list.append(torch.tanh(ra.fact_score(e_s, q, e, action[1], kg).detach()))
            # MA_value_list.append(ra.fact_score(e_s, q, e, action[1], kg).detach())

            kl_div = []
            for pi1_action_space, pi2_action_space in zip(db_outcomes, ra_db_outcomes):
                 kl_div.append(torch.nn.functional.kl_div(
                    ops.safe_log(pi2_action_space[1]),
                    pi1_action_space[1].detach(),
                    reduction='batchmean'
                ))
            # ratio = (ops.safe_log(action_prob) - ops.safe_log(ra_action_prob)).detach()
            # ratio = torch.nn.functional.l1_loss(ra_action_prob, action_prob.detach())
            # KL_loss = torch.mean(ops.safe_log(ra_action_prob / (action_prob + 1e8).detach()))
            # KL_loss = (ra_action_prob * (torch.log(ra_action_prob + 1e-8) - torch.log(action_prob.detach() + 1e-8))).mean()
            KL_loss = torch.stack(kl_div, 0).mean()
            # ratio = torch.clamp((ra_action_prob / (action_prob + 1e8)), 0.8, 1.2).detach()
            # ratio = torch.clamp((action_prob / (ra_action_prob + 1e-8)), 0.8, 1.2).detach()
            # ratio = torch.clamp(torch.exp(ops.safe_log(ra_action_prob) - ops.safe_log(action_prob.detach())), 0.8, 1.2)

            critic_loss = ra._critic_learn(RA_state, MA_state, ops.safe_log(ra_action_prob), ra_entropy, kg)
            # critic_loss = ra._critic_learn(state, next_state, ra_reward, done, ops.safe_log(space_prob), obs, kg)
            critic_loss_list.append(critic_loss)

            MA_reward_list.append(self.reward_fun(e_s, q, e_t, action[1]))
            # RA_reward_list.append((action[1] == e_t).float())
            # MA_value_list.append(ra.fact_score(e_s, q, e, action[1], kg).detach())

            MA_dones.append(MA_done)
            last_e = e

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        ra_hit_num = (ra_next_e == e_t).float()
        hit_num = (action[1] == e_t).float()
        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)
        with torch.no_grad():
            # (next_e == e_t).float().sum()
            # MA_value_list.append(torch.sigmoid(ra.fact_score(e_s, q, e_s, e_s, kg)))
            # MA_value_list.append(torch.tanh(ra.fact_score(e_s, q, e_s, e_s, kg)))
            MA_value_list.append(ra.fact_score(e_s, q, e_s, e_s, kg))
            for id in range(len(path_trace) - 1):
                item1 = path_trace[id]
                item2 = path_trace[id + 1]
                e = item1[1]
                next_e = item2[1]
                # MA_value_list.append(torch.sigmoid(ra.fact_score(e_s, q, e, next_e, kg)))
                # MA_value_list.append(torch.tanh(ra.fact_score(e_s, q, e, next_e, kg)))
                MA_value_list.append(ra.fact_score(e_s, q, e, next_e, kg))

                rel_sim = (rel_sim + self.fn_kg.get_relation_embeddings(item2[0]))
                # rel_sim_score = torch.sigmoid(torch.cosine_similarity(rel_sim, kg.get_relation_embeddings(q)))
                rel_sim_score = torch.cosine_similarity(rel_sim, self.fn_kg.get_relation_embeddings(q))
                rel_sim_list.append(rel_sim_score)

        # for i in range(len(path_trace) - 2):
        #     rel_sim += kg.get_relation_embeddings(path_trace[i + 1][0])
        #
        # rel_sim = torch.cosine_similarity(rel_sim, kg.get_relation_embeddings(q))
        # rel_sim = torch.stack(rel_sim_list, 0).detach()
        rel_reward, advantages = self.compute_advantage(MA_reward_list, MA_value_list, MA_dones, rel_sim_list)
        # entity_sim = torch.cosine_similarity(kg.get_entity_embeddings(pred_e2), kg.get_entity_embeddings(e_t))

        hit_rate = ra_hit_num.sum() / hit_num.sum()

        sim_rate = (action[1] == ra_next_e).float().sum() / len(hit_num)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components,
            'rel_reward': rel_reward,
            'advantages': advantages,
            'critic_loss': critic_loss_list,
            'entity_sim': rel_sim,
            'ra_value_list': MA_value_list,
            'sim_rate': sim_rate,
            'hit_rate': hit_rate,
            'cl_x': pn.path[-1][0][-1, :, :]
        }

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False):
        # MA_value_list, MA_reward_list, MA_dones, rel_sim_list = [], [], [], []
        kg, pn, ra = self.kg, self.mdl, self.ra
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, ra, self.num_rollout_steps, self.beam_size)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']

        # # print("search_trace shape:", search_trace[0])
        # rel_sim = self.fn_kg.get_relation_embeddings(search_trace[0][0])
        # for t in range(3):
        #     e = search_trace[t][1]
        #     if t == 2: pred_e2 = search_trace[t][1]
        #     # for action in search_trace[t + 1]:
        #     action = search_trace[t + 1]
        #     # print("action[1].size ", action[1].size(), "  e2.size ", e2.size())
        #     MA_done = (action[1] == whole_query[2]).float()
        #     MA_dones.append(MA_done)
        #     MA_value_list.append(ra.fact_score(whole_query[0], e, whole_query[1], action[1], kg).detach())
        #     MA_reward_list.append(self.reward_fun(whole_query[0], whole_query[1], whole_query[2], action[1]))
        #
        #     rel_sim = (rel_sim + self.fn_kg.get_relation_embeddings(action[0]))
        #     # rel_sim_score = torch.sigmoid(torch.cosine_similarity(rel_sim, kg.get_relation_embeddings(q)))
        #     rel_sim_score = torch.cosine_similarity(rel_sim, self.fn_kg.get_relation_embeddings(whole_query[1]))
        #     rel_sim_list.append(rel_sim_score)
        #
        # advantages = self.compute_advantage(MA_reward_list, MA_value_list, MA_dones, rel_sim_list)
        #
        # o_reward = []
        # for id in range(3):
        #     row_reward = []
        #     for ind in range(0, len(torch.stack(MA_dones, 0)[id]), 128):
        #         row_reward.append(torch.stack(MA_dones, 0)[id][ind : ind + 10])
        #     row_reward = torch.stack(row_reward, 0).reshape(-1)
        #     o_reward.append(row_reward)
        # np.savetxt('/data/weijinhui/KG/QSDR/test/origin_reward.csv', torch.stack(o_reward, 0).cpu().data.numpy(), fmt='%.6f', delimiter=',')
        #
        # # Compute discounted reward
        # final_reward = self.reward_fun(whole_query[0], whole_query[1], whole_query[2], pred_e2)
        #
        # cum_discounted_rewards = [0] * self.num_rollout_steps
        # cum_discounted_rewards[-1] = final_reward
        #
        # R = 0
        # for i in range(self.num_rollout_steps - 1, -1, -1):
        #     R = self.gamma * R + cum_discounted_rewards[i]
        #     cum_discounted_rewards[i] = R
        #     print(R)
        #
        # o_reward = []
        # for id in range(3):
        #     row_reward = []
        #     for ind in range(0, len(torch.stack(cum_discounted_rewards, 0)[id]), 128):
        #         row_reward.append(torch.stack(cum_discounted_rewards, 0)[id][ind : ind + 10])
        #     row_reward = torch.stack(row_reward, 0).reshape(-1)
        #     o_reward.append(row_reward)
        # np.savetxt('/data/weijinhui/KG/QSDR/test/origin_reward.csv', torch.stack(o_reward, 0).cpu().data.numpy(), fmt='%.6f', delimiter=',')
        #
        # # potential和origin间别差太大
        # potential_discount = 0.2
        # origin_discount = 0.8
        # for i in range(self.num_rollout_steps - 1, -1, -1):
        #     relative_potential = torch.relu(advantages[i])
        #     Rs = origin_discount * cum_discounted_rewards[i]
        #     Rp = potential_discount * relative_potential
        #     # cum_discounted_rewards[i] = done + (1 - done) * (Rs + Rp)
        #     # cum_discounted_rewards[i] = (rel_sim + relative_potential + cum_discounted_rewards[i] * (1 - done)) / 3 + done
        #     # cum_discounted_rewards[i] = (rel_sim + 1) * relative_potential + cum_discounted_rewards[i]
        #     # 早期需要探索，因此早期的探索占比高点
        #     cum_discounted_rewards[i] = (Rs + Rp)
        #     print(cum_discounted_rewards[i])
        #
        # new_reward = []
        # for id in range(3):
        #     row_reward = []
        #     for ind in range(0, len(torch.stack(cum_discounted_rewards, 0)[id]), 128):
        #         row_reward.append(torch.stack(cum_discounted_rewards, 0)[id][ind : ind + 10])
        #     row_reward = torch.stack(row_reward, 0).reshape(-1)
        #     new_reward.append(row_reward)
        # np.savetxt('/data/weijinhui/KG/QSDR/test/new_reward.csv', torch.stack(new_reward, 0).cpu().data.numpy(), fmt='%.6f', delimiter=',')

        # # verbose = True
        # if verbose:
        #     # print inference paths
        #     search_traces = beam_search_output['search_traces']
        #     output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
        #     for i in range(len(e1)):
        #         print("query(e_s, r_q, ?) : ", "(", kg.id2entity[int(e1[i])], ", ", kg.id2relation[int(r[i])], ", ?) The ans is : ", kg.id2entity[int(e2[i])])
        #         # print("search_traces len : ", len(search_traces))
        #         ans_count = 0
        #         for j in range(4):
        #             ind = i * output_beam_size + j
        #             if pred_e2s[i][j] == kg.dummy_e:
        #                 break
        #             search_trace = []
        #             for k in range(len(search_traces)):
        #                 search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
        #                 if k == len(search_traces) - 1 and int(search_traces[k][1][ind]) == int(e2[i]):
        #                     ans_count = ans_count + 1
        #             print('beam {}: score = {} \n<PATH> {}'.format(
        #                 j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))
        #         print("Correct ans nums : ", ans_count)
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss 一般到这里
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
