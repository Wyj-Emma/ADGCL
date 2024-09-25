import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

class Encode(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.C = nn.Linear(hidden, hidden // 2)
        self.P = nn.Linear(hidden, hidden // 2)

    def forward(self, emb):
        common = self.C(emb)
        private = self.P(emb)
        return common, private

class Decoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, hidden)

    def forward(self, s, p):
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons


class DAE(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoder = Encode(hidden)
        self.decoder = Decoder(hidden)

    def encode(self, emb):
        common, private = self.encoder(emb)
        return common, private

    def decode(self, s, p):
        return self.decoder(s, p)

    def forward(self, emb):
        common, private = self.encode(emb)
        recons = self.decode(common, private)

        return common, private, recons


class ADGCL(nn.Module):
    def __init__(self, data_config, args):
        super(ADGCL, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))],
            dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.temp = args.temp

        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg
        self.cor_reg = args.cor_reg
        self.ssl_reg = args.ssl_reg

        self.n_negs = args.n_negs
        self.ns = args.ns
        self.K = args.K
        self.gamma = args.gamma

        self.linear1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.normal_(self.linear1.bias, std=0.01)

        self.linear2 = nn.Linear(self.emb_dim, 1)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.normal_(self.linear2.bias, std=0.01)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

        self.user_gate = nn.Linear(self.emb_dim, self.emb_dim)
        self.item_gate = nn.Linear(self.emb_dim, self.emb_dim)
        self.pos_gate = nn.Linear(self.emb_dim, self.emb_dim)
        self.neg_gate = nn.Linear(self.emb_dim, self.emb_dim)

        self.dae_u = DAE(self.emb_dim)
        self.dae_v = DAE(self.emb_dim)

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.encoder = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.normal_(self.encoder.bias, std=0.01)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values,
                                             sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values,
                                                  self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0],
                                                  self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def label(self, lat1, lat2):
        lat = torch.cat([lat1, lat2], dim=-1)
        lat = self.leakyrelu(self.dropout(self.linear1(lat))) + lat1 + lat2
        ret = torch.reshape(self.sigmoid(self.dropout(self.linear2(lat))), [-1])
        return ret

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = F.normalize(head_embeddings)
        tail_embeddings = F.normalize(tail_embeddings)
        edge_alpha = self.label(head_embeddings, tail_embeddings)
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha,
                                             sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def inference(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        gnn_embeddings = []
        int_embeddings = []
        iaa_embeddings = []

        cor = 0

        for i in range(0, self.n_layers):
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)

            z = self.dae_u(u_embeddings)
            z_i = self.dae_v(i_embeddings)
            int_layer_embeddings = torch.concat([z[2] + u_embeddings, z_i[2] + i_embeddings], dim=0)
            cor += F.mse_loss(z[2], u_embeddings) + F.mse_loss(z_i[2], i_embeddings)

            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)

            iaa_layer_embeddings = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])

            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)

            all_embeddings.append(
                gnn_layer_embeddings + iaa_layer_embeddings + all_embeddings[i])

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        return gnn_embeddings, int_embeddings, iaa_embeddings, cor

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.n_users, self.n_items], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.n_users, self.n_items], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.n_users, self.n_items], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss

    def cal_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        u_e = user_gcn_emb
        pos_e = pos_gcn_embs
        neg_e = neg_gcn_embs.view(self.batch_size, self.K, -1)
        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)
        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        if self.ns == 'ns' and self.gamma > 0.:
            gate_pos = torch.sigmoid(self.item_gate(pos_gcn_embs) + self.user_gate(user_gcn_emb))
            gated_pos_e_r = pos_gcn_embs * gate_pos

            gate_neg = torch.sigmoid(self.neg_gate(neg_gcn_embs) + self.pos_gate(gated_pos_e_r).unsqueeze(1))
            gated_neg_e_r = neg_gcn_embs * gate_neg
            gated_neg_e_ir = neg_gcn_embs - gated_neg_e_r

            gated_neg_e_r = gated_neg_e_r.view(self.batch_size, self.K, -1)
            gated_neg_e_ir = gated_neg_e_ir.view(self.batch_size, self.K, -1)

            gated_neg_scores_r = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_r), axis=-1)
            gated_neg_scores_ir = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_ir), axis=-1)

            mf_loss += self.gamma * torch.mean(
                torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir).sum(dim=1)))
        return mf_loss

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates):
        s_e = user_gcn_emb[user]
        n_e = item_gcn_emb[neg_candidates]
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)
        indices = torch.max(scores, dim=1)[1].detach()
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()
        return item_gcn_emb[neg_item]


    def forward(self, users, pos_items, neg_items):
        users = torch.LongTensor(users).cuda()
        pos_items = torch.LongTensor(pos_items).cuda()
        neg_items = torch.LongTensor(neg_items).cuda()
        gnn_embeddings, int_embeddings, iaa_embeddings, cor = self.inference()
        neg_embeddings = []
        for k in range(self.K):
            neg_embeddings.append(self.negative_sampling(self.ua_embedding, self.ia_embedding,users,
                                                                 neg_items[:,k * self.n_negs: (k + 1) * self.n_negs]))
        neg_embeddings = torch.stack(neg_embeddings, dim=1)

        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        mf_loss = self.cal_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(
            2).pow(2))
        emb_loss = self.emb_reg * emb_loss
        cen_loss = self.cor_reg * cor
        cl_loss = self.ssl_reg * self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, iaa_embeddings)

        return mf_loss, emb_loss, cen_loss, cl_loss

    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(users).cuda()]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings


