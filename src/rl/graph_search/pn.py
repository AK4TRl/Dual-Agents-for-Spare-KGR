
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda
# import utils.ops as ops
# from utils.ops import var_cuda, zeros_var_cuda


class GraphSearchPolicy(nn.Module):
    def __init__(self, args, fn=None, fn_kg=None):
        super(GraphSearchPolicy, self).__init__()
        self.model = args.model
        self.relation_only = args.relation_only

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
        self.path = None

        # Set policy network modules
        self.define_modules()
        self.initialize_modules()

        # Fact network modules
        self.fn = fn
        self.fn_kg = fn_kg

    def transit(self, e, obs, ra, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        # def transit(self, e, obs, kg, ref_e, bucket, ra, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_action_space_bucketing: If set, group the action space of different nodes
            into buckets by their sizes.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        # db_action_spaces, db_references = bucket

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_embeddings(q)
        # REF_E = kg.get_entity_embeddings(ref_e)
        H = self.path[-1][0][-1, :, :]
        E = kg.get_entity_embeddings(e)

        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(e_s)
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:
            X = torch.cat([E, H, Q], dim=-1)

        # # #
        # G = []
        # for action_space_b, reference_b in zip(db_action_spaces, db_references):
        #     (r_space, e_space), action_mask = action_space_b
        #     A = self.get_action_embedding((r_space, e_space), kg)
        #     X_b = self.W_att2(X[reference_b, :]).unsqueeze(-1)
        #     alpha = torch.softmax(F.leaky_relu(torch.matmul(A, X_b)), 1)
        #     local_knowledge = torch.matmul(alpha.transpose(1, 2), A).squeeze(1)
        #     G.append(local_knowledge)
        # G = torch.cat(G, dim=0)[inv_offset]
        #
        # X = torch.cat([X, H, G], -1)
        # MLP
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        # all_relations_embeddings = kg.get_all_relation_embeddings()
        entities_probs = ra.critic.pred_entity(e_s, q, e, kg).detach()
        # entities_probs = F.sigmoid(all_entity_emb)
        # entities_probs = F.softmax(all_entity_emb)
        # pred_id = torch.multinomial(all_entity_emb, 1)
        # _, pred_id = torch.topk(entities_probs, 1, dim=-1)
        # pred_emb = kg.get_entity_embeddings(pred_id.squeeze(-1))
        # pred_emb = kg.get_entity_embeddings(pred_id)
        # rel_select_num = torch.sum(entities_probs > 0.95, -1)
        # TODO: 尝试拿到5个到10个的实体然后取嵌入均值。结果不太行，过多的信息导致噪声的增加
        # pred_id = torch.multinomial(entities_probs, 6).view(X2.size(0) * 6)
        pred_id = torch.multinomial(entities_probs, 1)
        # _, pred_id = torch.topk(entities_probs, 6, dim=-1)
        # pred_emb = kg.get_entity_embeddings(pred_id).sum(1)
        pred_emb = kg.get_entity_embeddings(pred_id)

        # pred_ent_info = self.W3(torch.cat([X2.unsqueeze(1).repeat_interleave(1, 1), pred_emb], -1))
        pred_ent_info = self.W3(pred_emb)

        # pred_ent_info = self.W4(S_E)
        relation_K = self.W4(kg.get_all_relation_embeddings())
        # relation_V = self.W5(kg.get_all_relation_embeddings())

        local_att = torch.matmul(pred_ent_info, relation_K.t())

        beta = torch.matmul(F.softmax(local_att, dim=-1), kg.get_all_relation_embeddings()).sum(1)
        # alpha = torch.bmm(F.softmax(local_att.permute(0, 2, 1), dim=-1), pred_emb).sum(1)
        alpha = self.W5(torch.cat([kg.get_all_relation_embeddings().unsqueeze(0).expand(pred_emb.size(0), -1, -1),
                                 pred_emb.expand(-1, kg.num_relations, -1)], -1).sum(1))
        # # local_att = F.softmax(torch.matmul(rel_info, kg.get_all_relation_embeddings().t()), -1) # 这里权重大小都差不多，感觉像是算不出来区别？
        # # local_rel = torch.matmul(local_att.transpose(1, 2), torch.cat([X2.unsqueeze(1).repeat_interleave(6, 1), pred_emb], -1)).mean(1)
        # local_rel = torch.matmul(local_att.transpose(1, 2), pred_emb).mean(1)
        # local_rel = torch.matmul(local_att.unsqueeze(-1), pred_emb.unsqueeze(1)).mean(1)
        # local_rel = torch.matmul(local_att, kg.get_all_relation_embeddings()).mean(1)    # 来到这里后，后续的计算使得所有状态都一样，感觉主要还是这里的问题？
        # V_A = self.W5(torch.cat([X2.unsqueeze(1).repeat_interleave(1, 1), pred_emb, beta], -1)).sum(1)
        V_A = self.W6(torch.cat([alpha, beta], -1))

        # gate = self.fusion_gate(torch.cat([X2, V_A], -1))
        # # gate = self.fusion_gate(X2)
        # X3 = X2 + gate * self.W7(V_A)

        # 原本这个就可以被看作为硬注意力，相当于将隐藏状态进行随机选择
        relation_att = torch.matmul(self.W_att(torch.cat([X2, V_A], -1)), kg.get_all_relation_embeddings().t())
        # relation_att = torch.matmul(self.W_att(X2), kg.get_all_relation_embeddings().t())
        # relation_att = torch.matmul(self.W_att(torch.cat([X2.unsqueeze(1).repeat_interleave(3, 1), pred_emb], -1)), kg.get_all_relation_embeddings().t()).mean(1)
        # B x |R|
        # Trick -> mask SIM relation
        relation_att = F.softmax(relation_att, dim=-1)
        # if last_step:
        #     # print("relation_att : ", relation_att[:7, :50].shape)
        #     relation_se = []
        #     entity_cur = []
        #     tmp0 = relation_att
        #     tmp_sum_0 = tmp0.sum(0)
        #     _, index = torch.topk(tmp_sum_0, 200, -1)
        #     # index = torch.tensor([113, 2, 259, 334, 379, 335, 380, 363, 187, 361, 69, 399, 107, 362, 97,  59,  91, 305, 303, 315])
        #     new_harvest_0 = []
        #     for item in index[:200]:
        #         new_harvest_0.append(tmp0[:, item].cpu().data.numpy())
        #         if len(relation_se) < 10:
        #             if kg.id2relation[int(item)] in ['DUMMY_RELATION', 'START_RELATION', 'NO_OP_RELATION']:
        #                 relation_se.append(kg.id2relation[int(item)])
        #             else:
        #                 relation_se.append(kg.id2relation[int(item)][8:])
        #     # print("relations index : ", index)
        #
        #     new_harvest = []
        #     tmp1 = torch.tensor(new_harvest_0)
        #     # print("tmp1.shape : ", tmp1.shape)
        #     tmp_sum_1 = tmp1.sum(0)
        #     _, index = torch.topk(tmp_sum_1, 200, -1)
        #     # index = torch. tensor([541, 2001,  165,  918,  263, 1761,  383, 1683, 2035,  137, 1384, 1046, 95,  531, 1152,  899, 1118,  178, 1497,  694])
        #     for item in range(0, tmp1.shape[1], 20):
        #     # for item in index[:200]:
        #         if len(entity_cur) < 10:
        #             new_harvest.append(tmp1[:, item].cpu().data.numpy())
        #             if kg.id2entity[int(e[item])] in ['DUMMY_ENTITY', 'NO_OP_ENTITY']:
        #                 entity_cur.append(kg.id2entity[int(e[item])])
        #             else:
        #                 entity_cur.append(kg.id2entity[int(e[item])][8:])
        #     # print("entities index : ", e[index])
        #
        #     # vegetables = ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley", "barley", "barley", "barley"]
        #
        #
        #     # # 计算均值
        #     # mean = torch.mean(torch.tensor(new_harvest), dim=1, keepdim=True)
        #     # # 计算标准差
        #     # std = torch.std(torch.tensor(new_harvest), dim=1, keepdim=True)
        #     # # 归一化
        #     # result = (torch.tensor(new_harvest) - mean) / std
        #     print("new_harvest.shape : ", torch.tensor(new_harvest).shape)
        #     new_harvest = torch.round(torch.tensor(new_harvest), decimals=2)
        #     harvest = new_harvest.data.cpu().numpy()
        #     plt.xticks(np.arange(10), labels=relation_se, rotation=45, rotation_mode="anchor", ha="right", fontsize=7)
        #     plt.yticks(np.arange(10), labels=entity_cur, fontsize=7)
        #     plt.title("No optimization in relations selection on NELL23k", fontsize=8)
        #     for i in range(10):
        #         for j in range(10):
        #             # print('i ', i)
        #             # print('j ', j)
        #             # print('harvest[i][j] ', harvest[i][j])
        #             text = plt.text(j, i, harvest[i, j], ha="center", va="center", color="black", fontsize=7)
        #     plt.imshow(harvest[:10, :10], cmap='Blues', aspect='equal', alpha=0.8)
        #     plt.tight_layout()
        #     plt.colorbar()
        #     plt.savefig('/data/weijinhui/KG/QSDR/test/test——ooooo.png', dpi=800, bbox_inches='tight')
        #     # plt.show()

        # entity_att = F.softmax(torch.matmul(rel_info, kg.get_all_entity_embeddings().t()), dim=-1)
        # local_ent = torch.matmul(entity_att, kg.get_all_entity_embeddings())

        # action_info = F.relu(self.W3(torch.cat([local_rel, local_ent], -1)))
        # action_info = self.W4(action_info)

        # X2 = self.W_att3(torch.cat([X2, local_rel], -1))

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

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

        if use_action_space_bucketing:
            """

            """
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets(e, obs, kg, relation_att=relation_att)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2[reference_b, :]
                # REF_E_b = REF_E[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
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
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2, action_space)
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        return db_outcomes, inv_offset, entropy, [db_action_spaces, db_references], H.detach()

    def initialize_path(self, init_action, kg):
        # [batch_size, action_dim]
        if self.relation_only_in_path:
            init_action_embedding = kg.get_relation_embeddings(init_action[0])
        else:
            init_action_embedding = self.get_action_embedding(init_action, kg)
        init_action_embedding.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        init_h = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        init_c = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        self.path = [self.path_encoder(init_action_embedding, (init_h, init_c))[1]]

    def update_path(self, action, kg, offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """
        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        if self.relation_only_in_path:
            action_embedding = kg.get_relation_embeddings(action[0])
        else:
            action_embedding = self.get_action_embedding(action, kg)
        if offset is not None:
            offset_path_history(self.path, offset)

        self.path.append(self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1])

    def get_dynamic_action_space(self, e_space, r_space, action_mask, e_b, relation_att):
        # bks -> bucket_size, ass -> action_space_size
        (bks, ass) = e_space.shape
        max_dynamic_action_size = 20
        dynamic_split_bound = 2
        avg_entity_per_relation = 5
        # relation_limt = rel_select_num_b.max() + 1
        # # 如果远远大于目前已有空间，那么已有空间存在的可能实体会较多，那么可以添加少一点
        # # 如果小于，那么就需要大一点
        # alpha = (ass / relation_limt)

        additional_action_space_size = min(int(ass / dynamic_split_bound) + 1, max_dynamic_action_size)
        # additional_action_space_size = min(int(alpha * ass) + 1, max_dynamic_action_size)
        # additional_action_space_size = min(int(relation_limt), max_dynamic_action_size)
        # additional_action_space_size = max_dynamic_action_size

        # 大于1的话，表示有足够的覆盖率；而小于1代表覆盖率不够，较为稀疏。
        # 覆盖率不够就只能加实体，而足够的覆盖率就加关系
        # if alpha > 1:
        #     additional_action_space_size = max_dynamic_action_size
        #     avg_entity_per_relation = 1
        # else:
        #     additional_action_space_size = min(int(alpha * ass) + 1, max_dynamic_action_size)
        #     avg_entity_per_relation = 5
        additional_action_space_size = 1

        additional_relation_size = int(additional_action_space_size / avg_entity_per_relation) + 1

        # bks x additional_relation_size
        # 在relation_att中抽取addtional_relation_size大小的关系
        relation_idx = torch.multinomial(relation_att, additional_relation_size)
        # bks x additional_relation_size x |E|
        # 让e_b在batch处重复additional_relation_size次，这里也是进行的对当前所处的实体以及关系进行的下一步预测，并改变预测结果的结构
        # 这个得到的是实体的概率
        # 后续再根据S的概率大小选择前K个可能性更大的
        S = self.fn.forward(e_b.repeat_interleave(additional_relation_size, dim=0), relation_idx.view(bks * additional_relation_size), self.fn_kg).view(bks, additional_relation_size, self.fn_kg.num_entities)

        # idx -> bks x additional_relation_size x self.avg_entity_per_relation
        # 选择得分前relation个作为新的预测关系
        _, idx = torch.topk(S, avg_entity_per_relation, dim=-1)
        # bks x (additional_relation_size * self.avg_entity_per_relation)
        # 利用新的关系以及实体作为新的动作以及关系空间
        new_r_space = relation_idx.repeat_interleave(avg_entity_per_relation, dim=1)
        new_e_space = idx.view(bks, -1)
        new_action_mask = torch.ones(bks, additional_relation_size * avg_entity_per_relation).cuda()
        e_space = torch.cat([e_space, new_e_space], dim=-1)
        r_space = torch.cat([r_space, new_r_space], dim=-1)
        action_mask = torch.cat([action_mask, new_action_mask], dim=-1)
        return e_space, r_space, action_mask

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
        db_action_spaces, db_references = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)
            for key in batch_ref:
                action_space = kg.action_space_buckets[key]
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
                l_batch_refs = batch_ref[key]
                g_bucket_ids = key2[l_batch_refs].tolist()
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_batch_refs]
                # rel_select_num_b = rel_select_num[l_batch_refs]

                e_space_b, r_space_b, action_mask_b = self.get_dynamic_action_space(e_space_b, r_space_b, action_mask_b,
                                                                                    e_b, relation_att[l_batch_refs])
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                q_b = q[l_batch_refs]
                e_t_b = e_t[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

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

        # # Prevent the agent from selecting the ground truth edge
        # ground_truth_edge_mask = self.get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg)
        # action_mask -= ground_truth_edge_mask
        # self.validate_action_mask(action_mask)

        # # Mask out false negatives in the final step
        # if last_step:
        #     false_negative_mask = self.get_false_negative_mask(e_space, e_s, q, e_t, kg)
        #     action_mask *= (1 - false_negative_mask)
        #     self.validate_action_mask(action_mask)

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

    def get_action_embedding(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph enviroment.
        """
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def define_modules(self):
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim
        hidden_dim = 100
        # self.W_b = nn.Linear(self.entity_dim, 100)
        # self.W_nei = nn.Linear(self.entity_dim, self.action_dim)
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        # self.W3 = nn.Linear(self.action_dim + self.entity_dim, hidden_dim)
        self.W3 = nn.Linear(self.entity_dim, hidden_dim)
        self.W4 = nn.Linear(self.relation_dim, hidden_dim)
        self.W5 = nn.Linear(self.relation_dim + self.entity_dim, self.entity_dim)
        self.W6 = nn.Linear(self.relation_dim + self.entity_dim, self.entity_dim)
        # self.W7 = nn.Linear(self.entity_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W3Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W4Dropout = nn.Dropout(p=self.ff_dropout_rate)
        # self.W_Q = nn.Linear(self.entity_dim, self.entity_dim)
        # self.W_K = nn.Linear(self.action_dim, self.entity_dim)
        # self.W3 = nn.Linear(self.entity_dim, self.action_dim)
        # self.W3Dropout = nn.Dropout(p=self.ff_dropout_rate)
        # self.layer_norm_1 = nn.LayerNorm(self.action_dim)
        self.W_att = nn.Linear(self.action_dim + self.entity_dim, self.relation_dim)
        # self.W_att = nn.Linear(self.action_dim, self.relation_dim)
        self.W_att2 = nn.Linear(self.action_dim + self.entity_dim, self.relation_dim)
        self.W_att3 = nn.Linear(self.action_dim + self.entity_dim, self.action_dim)

        # # 融合门控机制
        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(self.action_dim + self.entity_dim, self.action_dim),
        #     nn.ReLU(),
        #     nn.Linear( self.action_dim,  self.action_dim),
        #     nn.Sigmoid()
        # )

        if self.relation_only_in_path:
            self.path_encoder = nn.LSTM(input_size=self.relation_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)
        else:
            self.path_encoder = nn.LSTM(input_size=self.action_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)

    def initialize_modules(self):
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            # nn.init.xavier_uniform_(self.W3.weight)
            for name, param in self.path_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
