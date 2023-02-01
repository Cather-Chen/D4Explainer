import logging
import time
import os
import numpy as np
from datetime import datetime
import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
# from sample_ppgn_simple import sample_main, sample_testing
from explainers.diffusion.graph_utils import graph2tensor, tensor2graph, gen_list_of_data_single, generate_mask, gen_full
from explainers.diffusion.pgnn import Powerful
from explainers.base import Explainer

import wandb

# EPS = 1e3
def model_save(args, model, mean_train_loss, best_sparsity, mean_test_acc):
    to_save = {
        'model': model.state_dict(),
        "train_loss": mean_train_loss,
        "eval sparsity": best_sparsity,
        "eval acc": mean_test_acc
    }
    exp_dir = f'{args.root}/{args.dataset}/'
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    print(f"save model to {exp_dir}/best_model.pth")
    return

def loss_func_bce(score_list, groundtruth, sigma_list, mask, device, sparsity_level):
    '''
    params:
        score_list: [len(sigma_list)*bsz, N, N]
        groundtruth: [len(sigma_list)*bsz, N, N]
        mask:[len(sigma_list)*bsz, N, N]
    '''
    bsz = int(score_list.size(0) / len(sigma_list))
    num_node = score_list.size(-1)
    score_list = score_list * mask
    groundtruth = groundtruth * mask
    pos_weight = torch.full([num_node * num_node], sparsity_level).to(device)
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    score_list_ = torch.flatten(score_list, start_dim=1, end_dim=-1)
    groundtruth_ = torch.flatten(groundtruth, start_dim=1, end_dim=-1)
    loss_matrix = BCE(score_list_, groundtruth_)
    loss_matrix = loss_matrix.view(groundtruth.size(0), num_node, num_node)
    loss_matrix = loss_matrix * (
                1 - 2 * torch.tensor(sigma_list).repeat(bsz).unsqueeze(-1).unsqueeze(-1).expand(groundtruth.size(0), num_node, num_node).to(device)
                + 1.0 / len(sigma_list))
    weight_tensor = torch.where(groundtruth == 1, 1.0, 2.0).to(device)
        # torch.ones_like(groundtruth).to(device)  + groundtruth * 0.5
    loss_matrix = loss_matrix * mask
    loss_matrix = (loss_matrix + torch.transpose(loss_matrix, -2, -1)) / 2
    loss = torch.mean(loss_matrix)
    return loss

def sparsity(score, groundtruth, mask, threshold = 0.5):
    '''
    params:
        score: list of [bsz, N, N, 1], list: len(sigma_list),
        groundtruth: [bsz, N, N]
        mask: [bsz, N, N]
    '''
    score_tensor = torch.stack(score, dim=0).squeeze(-1)  # len_sigma_list, bsz, N, N]
    score_tensor = torch.mean(score_tensor, dim=0)  # [bsz, N, N]
    pred_adj = torch.where(torch.sigmoid(score_tensor) > threshold, 1, 0).to(groundtruth.device)
    # pred_adj = pred_adj * mask
    pred_adj = pred_adj * mask
    groundtruth = groundtruth * mask
    adj_diff = torch.abs(groundtruth-pred_adj)   #[bsz, N, N]
    num_edge_b = groundtruth.sum(dim=(1,2))
    adj_diff_ratio = adj_diff.sum(dim=(1,2)) / num_edge_b
    ratio_average = torch.mean(adj_diff_ratio)
    return ratio_average

def gnn_pred(graph_batch, graph_batch_sub, gnn_model, ds, task):
    # for module in gnn_model.modules():
    #     if isinstance(module, MessagePassing):
    #         module._explain = False
    #         module._edge_mask = None
    gnn_model.eval()
    if task == "nc":
        output_prob, _ = gnn_model.get_node_pred_subgraph(x=graph_batch.x, edge_index=graph_batch.edge_index,
                                                              mapping=graph_batch.mapping)
        output_prob_sub, _ = gnn_model.get_node_pred_subgraph(x=graph_batch_sub.x, edge_index=graph_batch_sub.edge_index,
                                                              mapping=graph_batch_sub.mapping)
    else:
        if ds == "mnist":
            output_prob, _ = gnn_model.get_pred(data=graph_batch)
            output_prob_sub, _ = gnn_model.get_pred(data=graph_batch_sub)
        else:
            output_prob, _ = gnn_model.get_pred(x=graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)
            output_prob_sub, _ = gnn_model.get_pred(x=graph_batch_sub.x, edge_index=graph_batch_sub.edge_index, batch=graph_batch_sub.batch)

    y_pred = output_prob.argmax(dim=-1)
    y_exp = output_prob_sub.argmax(dim=-1)
    return y_pred, y_exp


def loss_cf_exp(gnn_model, graph_batch, score, y_pred, y_exp, full_edge, mask, ds, task="nc"):
    #score: list of [bsz, N, N, 1], list: len(sigma_list); mask: [bsz, N, N]
    score_tensor = torch.stack(score, dim=0).squeeze(-1)  # len_sigma_list, bsz, N, N]
    score_tensor = torch.mean(score_tensor, dim=0).view(-1,1)  # [bsz*N*N,1]
    mask_bool = mask.bool().view(-1,1)
    edge_mask_full = score_tensor[mask_bool]  # [num_edge]
    assert edge_mask_full.size(0) == full_edge.size(1)
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # for module in gnn_model.modules():
    #     if isinstance(module, MessagePassing):
    #         module._explain = True
    #         module._apply_sigmoid = True
    #         module._edge_mask = edge_mask_full
    if task == "nc":
        output_prob_cont, output_repr_cont = gnn_model.get_pred_explain(x=graph_batch.x, edge_index=full_edge, edge_mask=edge_mask_full,
                                                                mapping=graph_batch.mapping)
    else:
        if ds == "mnist":
            graph_copy = graph_batch.clone()
            graph_copy.edge_index = full_edge
            output_prob_cont, output_repr_cont = gnn_model.get_pred_explain(data=graph_copy, edge_mask=edge_mask_full)
        else:
            output_prob_cont, output_repr_cont = gnn_model.get_pred_explain(x=graph_batch.x, edge_index=full_edge, edge_mask=edge_mask_full,
                                                                        batch=graph_batch.batch)
    n = output_repr_cont.size(-1)
    bsz = output_repr_cont.size(0)
    y_exp = output_prob_cont.argmax(dim=-1)
    inf_diag = torch.diag(-torch.ones((n)) / 0).unsqueeze(0).repeat(bsz, 1, 1).to(y_pred.device)
    neg_prop = (output_repr_cont.unsqueeze(1).expand(bsz, n, n) + inf_diag).logsumexp(-1) - output_repr_cont.logsumexp(-1).unsqueeze(1).repeat(1, n)
    loss_cf = criterion(neg_prop, y_pred)
    labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
    fid_drop = (1- output_prob_cont.gather(1, labels).view(-1)).detach().cpu().numpy()
    # fid_drop = (neg_prop.gather(1, labels).view(-1) - neg_prop.gather(1, labels).view(
    #     -1)).detach().cpu().numpy()
    fid_drop = np.mean(fid_drop)
    # assert graph_batch.self_y.shape[0] == y_pred_sub.size(0)  #bsz
    acc_cf = float(y_exp.eq(y_pred).sum().item() / y_pred.size(0)) #less, better
    return loss_cf, fid_drop, acc_cf



class DiffExplainer(Explainer):
    def __init__(self, device, gnn_model_path):
        super(DiffExplainer, self).__init__(device, gnn_model_path)
    def explain_graph_task(self, args, train_dataset, test_dataset):
        gnn_model = self.model.to(args.device)
        model = Powerful(args).to(args.device)
        self.train(args, model, gnn_model, train_dataset, test_dataset)

    def train(self, args, model, gnn_model, train_dataset, test_dataset):
        best_sparsity = np.inf
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        noise_list = args.noise_list
        # wandb.watch(model, criterion=torch.nn.NLLLoss(), log_freq=1)
        for epoch in range(args.epoch):
            print(f"start epoch {epoch}")
            train_losses = []
            train_loss_dist = []
            train_loss_cf = []
            train_acc = []
            train_fid = []
            train_sparsity = []
            train_remain = []
            t_start = time.time()
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
            for i, graph in enumerate(train_loader):
                graph.to(args.device)
                train_adj_b, train_x_b = graph2tensor(graph, device=args.device)
                # train_adj_b: [bsz, N, N]; train_x_b: [bsz, N, C]
                sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length)) \
                    if noise_list is None else noise_list
                train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32) #[bsz, N]
                # all nodes that are not connected with others
                if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
                train_x_b, train_ori_adj_b, train_node_flag_sigma, train_noise_adj_b, noise_diff =  \
                    gen_list_of_data_single(train_x_b, train_adj_b, train_node_flag_b, sigma_list, args)
                #train_ori_adj_b: target adjacency matrix, [len(sigma_list) * batch_size, N, N]
                #train_noise_adj_b: noisy adjacency matrxi, [len(sigma_list) * batch_size, N, N]
                optimizer.zero_grad()
                train_noise_adj_b_chunked = train_noise_adj_b.chunk(len(sigma_list), dim=0)
                train_x_b_chunked = train_x_b.chunk(len(sigma_list), dim=0)
                train_node_flag_sigma = train_node_flag_sigma.chunk(len(sigma_list), dim=0)
                score = []
                masks = []
                for i, sigma in enumerate(sigma_list):
                    mask = generate_mask(train_node_flag_sigma[i])
                    score_batch = model(A=train_noise_adj_b_chunked[i].to(args.device),
                                        node_features=train_x_b_chunked[i].to(args.device), mask=mask.to(args.device),
                                        noiselevel=sigma)   # [bsz, N, N, 1]
                    score.append(score_batch)
                    masks.append(mask)
                graph_batch_sub = tensor2graph(graph, score, mask)
                y_pred, y_exp = gnn_pred(graph,  graph_batch_sub, gnn_model, ds=args.dataset, task=args.task)
                full_edge_index = gen_full(graph.batch, mask)
                score_b = torch.cat(score, dim=0).squeeze(-1).to(args.device) # [len(sigma_list)*bsz, N, N]
                masktens = torch.cat(masks, dim=0).to(args.device) # [len(sigma_list)*bsz, N, N]
                modif_r = sparsity(score, train_adj_b, mask)
                remain_r = sparsity(score, train_adj_b, train_adj_b)
                loss_cf, fid_drop, acc_cf = loss_cf_exp(gnn_model, graph, score, y_pred, y_exp, full_edge_index, mask, ds=args.dataset, task=args.task)
                loss_dist = loss_func_bce(score_b, train_ori_adj_b, sigma_list, masktens, device=args.device, sparsity_level=args.sparsity_level)
                loss = loss_dist + args.alpha_cf * loss_cf
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_loss_dist.append(loss_dist.item())
                train_loss_cf.append(loss_cf.item())
                train_acc.append(acc_cf)
                train_fid.append(fid_drop)
                train_sparsity.append(modif_r.item())
                train_remain.append(remain_r.item())
            scheduler.step(epoch)
            mean_train_loss = np.mean(train_losses)
            mean_train_loss_dist = np.mean(train_loss_dist)
            mean_train_loss_cf = np.mean(train_loss_cf)
            mean_train_acc = np.mean(train_acc)
            mean_train_fidelity = np.mean(train_fid)
            mean_train_sparsity = np.mean(train_sparsity)
            mean_remain_rate = np.mean(train_remain)
            wandb_dict = {'Train Epoch': epoch, 'Time Used': time.time() - t_start, 'Total Loss': mean_train_loss,
                          'Distribution Loss': mean_train_loss_dist, 'Counterfactual Loss': mean_train_loss_cf,
                          'Train Acc':mean_train_acc, 'fidelity':mean_train_fidelity, 'Edit Ratio': mean_train_sparsity,
                          'Remain Ratio': mean_remain_rate}
            # wandb.log(wandb_dict)
            print((f'Training Epoch: {epoch} | '
                             f'training loss: {mean_train_loss} | '
                             f'training fidelity drop: {mean_train_fidelity} | '
                             f'training acc: {mean_train_acc} | '
                             f'training average modification: {mean_train_sparsity} | '))
            # evaluation
            if (epoch + 1) % args.verbose == 0:
                test_losses = []
                test_loss_dist = []
                test_loss_cf = []
                test_acc = []
                test_fid = []
                test_sparsity = []
                test_remain = []
                test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batchsize, shuffle=False)
                model.eval()
                for graph in test_loader:
                    graph.to(args.device)
                    test_adj_b, test_x_b = graph2tensor(graph, device=args.device)
                    test_x_b = test_x_b.to(args.device)
                    test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                    sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length)) \
                        if noise_list is None else noise_list
                    if isinstance(sigma_list, float):
                        sigma_list = [sigma_list]
                    test_x_b, test_ori_adj_b, test_node_flag_sigma, test_noise_adj_b, noise_diff = \
                        gen_list_of_data_single(test_x_b, test_adj_b, test_node_flag_b, sigma_list, args)
                    with torch.no_grad():
                        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
                        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
                        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
                        score = []
                        masks = []
                        for i, sigma in enumerate(sigma_list):
                            mask = generate_mask(test_node_flag_sigma[i])
                            score_batch = model(A=test_noise_adj_b_chunked[i].to(args.device),
                                                node_features=test_x_b_chunked[i].to(args.device), mask=mask.to(args.device),
                                                noiselevel=sigma).to(args.device) # [bsz, N, N, 1]
                            masks.append(mask)
                            score.append(score_batch)
                        graph_batch_sub = tensor2graph(graph, score, mask)
                        y_pred, y_exp = gnn_pred(graph, graph_batch_sub, gnn_model, ds=args.dataset, task=args.task)
                        full_edge_index = gen_full(graph.batch, mask)
                        score_b = torch.cat(score, dim=0).squeeze(-1).to(args.device)
                        masktens = torch.cat(masks, dim=0).to(args.device)
                        modif_r = sparsity(score, test_adj_b, mask)
                        reamin_r = sparsity(score, test_adj_b, test_adj_b)
                        loss_cf, fid_drop, acc_cf = loss_cf_exp(gnn_model, graph, score, y_pred,y_exp, full_edge_index, mask, ds=args.dataset, task=args.task)
                        loss_dist = loss_func_bce(score_b, test_ori_adj_b, sigma_list, masktens, device=args.device, sparsity_level=args.sparsity_level)
                        loss = loss_dist + args.alpha_cf * loss_cf
                        test_losses.append(loss.item())
                        test_loss_dist.append(loss_dist.item())
                        test_loss_cf.append(loss_cf.item())
                        test_acc.append(acc_cf)
                        test_fid.append(fid_drop)
                        test_sparsity.append(modif_r.item())
                        test_remain.append(reamin_r.item())
                mean_test_loss = np.mean(test_losses)
                mean_test_loss_dist = np.mean(test_loss_dist)
                mean_test_loss_cf = np.mean(test_loss_cf)
                mean_test_acc = np.mean(test_acc)
                mean_test_fid = np.mean(test_fid)
                mean_test_sparsity = np.mean(test_sparsity)
                mean_test_reamin = np.mean(test_remain)
                wandb_dict = {'Test Epoch': epoch, 'Time Used (Test)': time.time() - t_start, 'Total Loss (Test)': mean_test_loss,
                              'Distribution Loss (Test)': mean_test_loss_dist, 'Counterfactual Loss (Test)': mean_test_loss_cf,
                              'Test Acc': mean_test_acc, "Edit Ratio (Test)": mean_test_sparsity, 'Test Fidelity': mean_test_fid,
                              'Test Remain Rate': mean_test_reamin}
                # wandb.log(wandb_dict)

                print((f'Evaluation Epoch: {epoch} | '
                             f'test loss: {mean_test_loss} | '
                             f'test acc: {mean_test_acc} | '
                             f'test fidelity drop: {mean_test_fid} | '
                             f'test average modification: {mean_test_sparsity} | '
                             f'test remain rate: {mean_test_reamin} | '))
                if mean_test_sparsity < best_sparsity:
                    best_sparsity = mean_test_sparsity
                    model_save(args, model, mean_train_loss, best_sparsity, mean_test_acc)
