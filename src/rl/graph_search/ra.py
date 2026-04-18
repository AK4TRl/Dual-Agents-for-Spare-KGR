"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Graph Search Policy Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

# import utils.ops as ops
# from utils.ops import var_cuda, zeros_var_cuda

import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda

class Critic(nn.Module):
    def __init__(self, entity_dim, relation_dim, num_entities):
        super(Critic, self).__init__()
        # # Q1 network
        # self.l1 = nn.Linear(obs_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 1)
        # # Q2 network
        # self.l4 = nn.Linear(obs_dim + action_dim, 256)
        # self.l5 = nn.Linear(256, 256)
        # self.l6 = nn.Linear(256, 1)

        self.W1 = nn.Linear(entity_dim + relation_dim, relation_dim)
        # self.W1 = nn.Linear(relation_dim, relation_dim)

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.action_dim = self.entity_dim + self.relation_dim
        self.num_entities = num_entities

        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        # self.HypER_loss = torch.nn.BCELoss()

        # x = F.conv2d(x, k, groups=e1.size(0))
        self.in_channels = 1
        self.out_channels = 32
        self.filt_h = 1
        self.filt_w = 9

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.num_entities)))
        fc_length = (1 - self.filt_h + 1) * (self.entity_dim - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, self.entity_dim)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(self.relation_dim, fc1_length)

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.action_dim = self.entity_dim + self.relation_dim


    def pred_entity(self, e_s, q_s, e_c, kg): 
        e_s = kg.get_entity_embeddings(e_s)
        e_c = kg.get_entity_embeddings(e_c)
        r = kg.get_relation_embeddings(q_s)

        e_c = self.W1(torch.cat([e_s, e_c], -1))
        e1 = e_c.view(-1, 1, 1, kg.entity_embeddings.weight.size(1))
        e2 = kg.get_all_entity_embeddings()

        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        all_entity_prob = torch.mm(x, e2.transpose(1, 0))
        all_entity_prob += self.b.expand_as(all_entity_prob)
        S = torch.sigmoid(all_entity_prob)

        return S


    def forward_fact(self, e_s, q_s, e_c, e_2, kg):
        e_s = kg.get_entity_embeddings(e_s)
        e_c = kg.get_entity_embeddings(e_c)
        r = kg.get_relation_embeddings(q_s)

        e_c = self.W1(torch.cat([e_s, e_c], -1))
        e1 = e_c.view(-1, 1, 1, kg.entity_embeddings.weight.size(1))
        e2 = kg.get_entity_embeddings(e_2)

        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.matmul(x.unsqueeze(1), e2.unsqueeze(2)).squeeze(2)
        x += self.b[e_2].unsqueeze(1)
        S = torch.sigmoid(x)
        # S = torch.tanh(x)

        return S

class Actor(nn.Module):
    def __init__(self, entity_dim, relation_dim, history_dim, ff_dropout_rate, action_dropout_rate):
        super(Actor, self).__init__()

        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.action_dim = self.entity_dim + self.relation_dim
        self.ff_dropout_rate = ff_dropout_rate
        self.history_dim = history_dim
        self.action_dropout_rate = action_dropout_rate

        self.W1 = nn.Linear(self.action_dim + self.history_dim, self.action_dim)
        # self.W1 = nn.Linear(self.action_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W_att = nn.Linear(self.action_dim, self.entity_dim)

        self.path = None

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

    def RA_transit(self, action_space, X2, kg):
        ((r_space, e_space), action_mask) = action_space
        ri = kg.get_relation_embeddings(r_space)
        ei = kg.get_entity_embeddings(e_space)
        A = torch.cat([ri, ei], -1)
        dist = A @ torch.unsqueeze(X2, 2)
        action_dist = F.softmax(dist.squeeze(-1) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
        return action_dist, ops.entropy(action_dist)

    def select_action(self, e_t, H, r_q, REAL_E, kg, bucket, merge_aspace_batching_outcome):
        db_action_spaces, db_references = bucket
        X = torch.cat([e_t, H, r_q], -1)
        # MLP
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        references = []
        db_outcomes = []
        entropy_list = []
        for action_space_b, reference_b in zip(db_action_spaces, db_references):
            X2_b = X2[reference_b, :]
            action_dist_b, entropy_b = self.RA_transit(action_space_b, X2_b, kg)
            references.extend(reference_b)
            db_outcomes.append((action_space_b, action_dist_b))
            entropy_list.append(entropy_b)
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        entropy = torch.cat(entropy_list, dim=0)[inv_offset]

        if merge_aspace_batching_outcome:
            db_action_dist = []
            for _, action_dist in db_outcomes:
                db_action_dist.append(action_dist)
            action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
            action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        # ra_db_outcomes, ra_entropy, ra_inv_offset
        ra_sample_outcome = self.sample_action(db_outcomes, inv_offset)
        ra_action = ra_sample_outcome['action_sample']
        # ra.actor.update_path(ra_action, kg)
        ra_action_prob = ra_sample_outcome['action_prob']
        ra_next_e = ra_action[1]

        return ra_action_prob, ra_next_e, db_outcomes, entropy

class Representive_Agent(object):
# class Representive_Agent(nn.Module):
    def __init__(self, args, fn=None, fn_kg=None):
        super(Representive_Agent, self).__init__()
        self.model = args.model
        self.relation_only = args.relation_only

        self.beta = args.beta

        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        if self.relation_only:
            self.action_dim = args.relation_dim
        else:
            self.action_dim = args.entity_dim + args.relation_dim
        self.ff_dropout_rate = args.ff_dropout_rate
        self.rnn_dropout_rate = args.rnn_dropout_rate
        self.action_dropout_rate = args.action_dropout_rate

        self.xavier_initialization = args.xavier_initialization

        self.relation_only_in_path = args.relation_only_in_path

        self.emb_dropout_rate = args.emb_dropout_rate

        self.num_entities = fn_kg.num_entities
        self.num_relations = fn_kg.num_relations

        self.args = args
        self.device = torch.device(args.gpu)
        self.action_dropout_rate = args.action_dropout_rate

        self.actor = Actor(self.entity_dim, self.relation_dim, self.history_dim, self.ff_dropout_rate, self.action_dropout_rate).to(self.device)
        self.actor_optimizer = optim.Adam(params=list(self.actor.parameters()), lr=1e-3)

        self.critic = Critic(self.entity_dim, self.relation_dim, self.num_entities).to(self.device)
        # self.target_critic = copy.deepcopy(self.critic)
        # list(self.critic.parameters()) +
        self.critic_optimizer = optim.Adam(params=list(self.critic.parameters()), lr=1e-3)

        # self.ra_optimizer = optim.Adam(params=list(self.parameters()), lr=1e-3)

        # self.critic_optimizer = optim.Adam(params=list(self.parameters()), lr=1e-4)

        # Set policy network modules
        # self.define_modules()
        # self.print_all_model_parameters()
        # self.initialize_modules()

        # Fact network modules
        self.fn = fn
        self.fn_kg = fn_kg

    # def print_all_model_parameters(self):
    #     print('\nRA Model Parameters')
    #     print('--------------------------')
    #     for name, param in self.named_parameters():
    #         print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
    #     param_sizes = [param.numel() for param in self.parameters()]
    #     print('Total # parameters = {}'.format(sum(param_sizes)))
    #     print('--------------------------')
    #     print()

    def ra_reward(self, e_s, e, q, action, e_t, kg):

        pred_e = action
        real_reward = self.critic.forward_fact(e_s, e, q, pred_e, kg).squeeze(1).detach()
        real_reward_mask = (real_reward > 0).float()
        real_reward *= real_reward_mask
        # binary_reward = torch.gather(e_t, 1, pred_e.unsqueeze(-1)).squeeze(-1).float()
        binary_reward = (pred_e == e_t).float()
        reward = binary_reward + (1 - binary_reward) * real_reward

        return reward

    def fact_score(self, e_s, r, e, pred_e, kg):
            return self.critic.forward_fact(e_s, r, e, pred_e, kg).squeeze(-1)

    def _critic_learn(self, RA_state, MA_state, log_space_probs, ra_entropy, kg):

        ops.detach_module(kg)
        MA_reward, action, MA_done = MA_state
        e_s, q, e, e_t, last_e, RA_reward, RA_next_e, RA_done = RA_state
        real_rel, real_ent = action
        current_q = self.fact_score(e_s, q, last_e, e, kg)
        gamma = 1.0
        with torch.no_grad():
            RA_value = self.fact_score(e_s, q, e, RA_next_e, kg)
            MA_value = self.fact_score(e_s, q, e, real_ent, kg)     # 目标target，相当于next state和next action
            RA_target = (RA_reward + gamma * RA_value) / 2 # * (1 - RA_done)
            MA_target = (MA_reward + gamma * MA_value) / 2 # * (1 - MA_done)
            td_delta = RA_target - current_q

        critic_loss = (F.mse_loss(current_q, RA_target) + F.mse_loss(current_q, MA_target)) / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # if last_step:
        actor_loss = torch.mean(-log_space_probs * td_delta) # + KL_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ops.activate_module(kg)

        return torch.mean(critic_loss)

    def perceive(self, e, H, refer_e, obs, kg, bucket, merge_aspace_batching_outcome=False):
        '''
            params:
            e: current entity
            obs: 观测值，包括原始节点、查询关系、目标节点以及上一步到达当前实体的关系
        '''
        ops.detach_module(kg)

        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        db_action_spaces, db_references = bucket
        e_cur = kg.get_entity_embeddings(e)
        q_r = kg.get_relation_embeddings(q)
        REAL_E = kg.get_entity_embeddings(refer_e)
        ra_action_prob, ra_next_e, db_outcomes, entropy = self.actor.select_action(e_cur, H, q_r, REAL_E, kg, bucket, merge_aspace_batching_outcome)

        ops.activate_module(kg)

        return ra_action_prob, ra_next_e, db_outcomes, entropy

    def get_action_space_in_buckets(self, e, obs, kg, collapse_entities=False, relation_att=None):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        :return db_action_spaces:
            [((r_space_b0, r_space_b0), action_mask_b0),
             ((r_space_b1, r_space_b1), action_mask_b1),
             ...
             ((r_space_bn, r_space_bn), action_mask_bn)]

            A list of action space tensor representations grouped in n buckets, s.t.
            r_space_b0.size(0) + r_space_b1.size(0) + ... + r_space_bn.size(0) = e.size(0)

        :return db_references:
            [l_batch_refs0, l_batch_refs1, ..., l_batch_refsn]
            l_batch_refsi stores the indices of the examples in bucket i in the current batch,
            which is used later to restore the output results to the original order.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        assert(len(e) == len(last_r))
        assert(len(e) == len(e_s))
        assert(len(e) == len(q))
        assert(len(e) == len(e_t))
        db_action_spaces, db_references, episode_log_probabilities, refer_e, E_t, next_e, next_r = [], [], [], [], [], [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            episode_entropies = []
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)    # 记录实体的序号，将序号和bucket索引结合起来
            for key in batch_ref:           # 迭代key，也就是bucket索引
                action_space = kg.action_space_buckets[key] # 把索引为key的action_space拿出来
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref[key]               # 把索引为key的所有节点序号拿出来
                g_bucket_ids = key2[l_batch_refs].tolist()  # Key2记录的是对应节点action space的id，将对应的节点序号的动作空间拿出来
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_batch_refs]
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                q_b = q[l_batch_refs]
                e_t_b = e_t[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]  # 将上面信息打包成obs

                # H_b = H[l_batch_refs]

                # 在这里进行动态空间添加
                e_space_b, r_space_b, action_mask_b = self.get_dynamic_action_space(e_space_b, r_space_b, action_mask_b,
                                                                                    e_b, relation_att[l_batch_refs], kg)

                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)

                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def apply_action_masks(self, action_space, e, obs, kg):
        (r_space, e_space), action_mask = action_space
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Prevent the agent from selecting the ground truth edge
        if len(e_t.shape) == 1 or e_t.shape[-1] == 1:
            ground_truth_edge_mask = self.get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg)
            action_mask -= ground_truth_edge_mask
            self.validate_action_mask(action_mask)
        else:
            ground_truth_edge_mask = self.get_ground_truth_edge_mask_multi(e, r_space, e_space, e_s, q, e_t, kg)
            action_mask -= ground_truth_edge_mask
            self.validate_action_mask(action_mask)

        # TODO: Change it to the DacKGR
        # # Prevent the agent from selecting the ground truth edge
        # ground_truth_edge_mask = self.get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg)
        # action_mask -= ground_truth_edge_mask
        # self.validate_action_mask(action_mask)
        #
        # # Mask out false negatives in the final step
        # if last_step:
        #     false_negative_mask = self.get_false_negative_mask(e_space, e_s, q, e_t, kg)
        #     action_mask *= (1 - false_negative_mask)
        #     self.validate_action_mask(action_mask)
        #
        # # Prevent the agent from stopping in the middle of a path
        # # stop_mask = (last_r == NO_OP_RELATION_ID).unsqueeze(1).float()
        # # action_mask = (1 - stop_mask) * action_mask + stop_mask * (r_space == NO_OP_RELATION_ID).float()
        # # Prevent loops
        # # Note: avoid duplicate removal of self-loops
        # # seen_nodes_b = seen_nodes[l_batch_refs]
        # # loop_mask_b = (((seen_nodes_b.unsqueeze(1) == e_space.unsqueeze(2)).sum(2) > 0) *
        # #      (r_space != NO_OP_RELATION_ID)).float()
        # # action_mask *= (1 - loop_mask_b)
        return (r_space, e_space), action_mask

    def get_ground_truth_edge_mask(self, e, r_space, e_space, e_s, q, e_t, kg):
        ground_truth_edge_mask = \
            ((e == e_s).unsqueeze(1) * (r_space == q.unsqueeze(1)) * (e_space == e_t.unsqueeze(1)))
        inv_q = kg.get_inv_relation_id(q)
        inv_ground_truth_edge_mask = \
            ((e == e_t).unsqueeze(1) * (r_space == inv_q.unsqueeze(1)) * (e_space == e_s.unsqueeze(1)))
        return ((ground_truth_edge_mask + inv_ground_truth_edge_mask) * (e_s.unsqueeze(1) != kg.dummy_e)).float()

    def get_ground_truth_edge_mask_multi(self, e, r_space, e_space, e_s, q, e_t, kg):
        ans_1 = (torch.gather(e_t, 1, e_space) == 1)
        ans_2 = (torch.gather(e_t, 1, e.unsqueeze(-1)) == 1)

        ground_truth_edge_mask = \
            ((e == e_s).unsqueeze(1) * (r_space == q.unsqueeze(1)) * ans_1)
        inv_q = kg.get_inv_relation_id(q)
        inv_ground_truth_edge_mask = \
            (ans_2 * (r_space == inv_q.unsqueeze(1)) * (e_space == e_s.unsqueeze(1)))
        return ((ground_truth_edge_mask + inv_ground_truth_edge_mask) * (e_s.unsqueeze(1) != kg.dummy_e)).float()

    def get_answer_mask(self, e_space, e_s, q, kg):
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_object_vectors
        else:
            answer_vectors = kg.train_object_vectors
        answer_masks = []
        for i in range(len(e_space)):
            _e_s, _q = int(e_s[i]), int(q[i])
            if not _e_s in answer_vectors or not _q in answer_vectors[_e_s]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e_s][_q]
            answer_mask = torch.sum(e_space[i].unsqueeze(0) == answer_vector, dim=0).long()
            answer_masks.append(answer_mask)
        answer_mask = torch.cat(answer_masks).view(len(e_space), -1)
        return answer_mask

    def get_false_negative_mask(self, e_space, e_s, q, e_t, kg):
        answer_mask = self.get_answer_mask(e_space, e_s, q, kg)
        # This is a trick applied during training where we convert a multi-answer predction problem into several
        # single-answer prediction problems. By masking out the other answers in the training set, we are forcing
        # the agent to walk towards a particular answer.
        # This trick does not affect inference on the test set: at inference time the ground truth answer will not
        # appear in the answer mask. This can be checked by uncommenting the following assertion statement.
        # Note that the assertion statement can trigger in the last batch if you're using a batch_size > 1 since
        # we append dummy examples to the last batch to make it the required batch size.
        # The assertion statement will also trigger in the dev set inference of NELL-995 since we randomly
        # sampled the dev set from the training data.
        # assert(float((answer_mask * (e_space == e_t.unsqueeze(1)).long()).sum()) == 0)
        false_negative_mask = (answer_mask * (e_space != e_t.unsqueeze(1)).long()).float()
        return false_negative_mask

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert (action_mask_min == 0 or action_mask_min == 1)
        assert (action_mask_max == 0 or action_mask_max == 1)
