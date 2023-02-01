import os
import os.path as osp
import pdb
import time
from datetime import datetime
import numpy as np

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from explainers.base import Explainer
from explainers.gflow.agent import create_agent
from explainers.gflow.mdp import create_mdps
from explainers.gflow.sampler import create_samplers  # , stop_samplers_and_join
import networkx as nx
import matplotlib.pyplot as plt

EPS = 1e-15


class GFlowExplainer(Explainer):

    def __init__(self, device, gnn_model_path):
        super(GFlowExplainer, self).__init__(device, gnn_model_path)
        self.random_state = np.random.RandomState(int(time.time()))

    def explain_graph_task(self, args, train_dataset, eval_dataset, model=None,
                           save=True, **kwargs):

        self.cf_flag = args.cf_flag
        exp_dir = None
        if save:
            exp_dir = f'{args.root}/{args.dataset}/'
            os.makedirs(exp_dir, exist_ok=True)
        gnn_model = self.model if model is None else model
        gnn_model = gnn_model.to(self.device)
        agent = create_agent(args, device=self.device)
        print(f"Create mdps...")
        mdps = create_mdps(train_dataset, gnn_model, device=self.device)
        print(f"Create samplers...")
        samplers = create_samplers(args.n_thread, args.mbsize, mdps, agent, args.sampling_model_prob)
        print(f"Start training...")
        self.train_agent(args, agent, train_dataset, eval_dataset, gnn_model, samplers, exp_dir)

    def train_agent(self, args, agent, train_dataset, eval_dataset,
                    gnn_model, samplers, exp_dir):
        '''Train the RL agent'''

        for param in gnn_model.parameters():
            param.requires_grad = False
        torch.autograd.set_detect_anomaly(True)

        # 1. initialization of variables, optimizer, and sampler
        debug_no_threads = False
        mbsize = args.mbsize
        log_reg_c = args.log_reg_c
        leaf_coef = args.leaf_coef
        balanced_loss = args.balanced_loss
        clip_loss = torch.tensor([args.clip_loss], device=self.device).to(args.floatX)

        last_losses = []
        train_infos, test_infos = [], []
        time_last_check = time.time()
        datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

        optimizer = torch.optim.Adam(agent.parameters(),
                                     args.learning_rate,
                                     weight_decay=args.weight_decay,
                                     betas=(args.opt_beta, args.opt_beta2),
                                     eps=args.opt_epsilon
                                     )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.num_steps // 3)
        # for i in range(args.num_steps):
        for i in range(len(samplers)):
            print(f"begin epoch {i}")
            epoch_start = time.time()
            # 2. Sample trajectories on MDP
            lr = scheduler.optimizer.param_groups[0]['lr']
            # sampler_idx = self.random_state.randint(0, len(samplers), 1).item()

            sampler = samplers[i]

            if not debug_no_threads:
                samples = sampler()
                for thread in sampler.sampler_threads:
                    if thread.failed:
                        sampler.stop_samplers_and_join()
                        pdb.post_mortem(thread.exception.__traceback__)
                        return
                (p_graph, p_state), pb, action, reward, (s_graph, s_state), done = samples
            else:
                (p_graph, p_state), pb, action, reward, (s_graph, s_state), done = sampler.sample2batch(
                    sampler.sample_multiple(mbsize, PP=False))
            done = done.float()
            # if time.time()-epoch_start > 60:
            #     G = to_networkx(sampler.graph)
            #     nx.draw_networkx(G)
            #     plt.show()

            # Since we sampled 'mbsize' trajectories, we're going to get
            # roughly mbsize * H (H is variable)  transitions
            ntransitions = reward.shape[0]
            # 3. Forward the trajectories in agent to compute the flows
            # state outputs
            edge_out_s, graph_out_s = agent(s_graph, s_state, sampler.graph, gnn_model, actions=None)
            # parents of the state outputs
            edge_out_p, graph_out_p = agent(p_graph, p_state, sampler.graph, gnn_model, actions=action)
            # index parents by their corresponding actions
            qsa_p = agent.index_output_by_action(p_graph, edge_out_p, graph_out_p, action)
            # then sum the parents' contribution, this is the inflow
            exp_inflow = (torch.zeros((ntransitions,), device=self.device)
                          .index_add_(0, pb, torch.exp(qsa_p)))  # pb is the parents' batch index
            inflow = torch.log(exp_inflow + log_reg_c)

            # sum the state's Q(s,a), this is the outflow
            edge_out_s = torch.cat([edge_out.view(1, -1).to(self.device) for edge_out in edge_out_s], dim=0)
            exp_outflow = agent.sum_output(s_graph, torch.exp(edge_out_s), 0)
            # include reward and done multiplier, then take the log
            # we're guarenteed that reward > 0 iff done = 1, so the log always works
            outflow_plus_r = torch.log(log_reg_c + reward + exp_outflow * (1 - done))

            # 4. Compute flow loss and backward
            losses = (inflow - outflow_plus_r).pow(2)

            if clip_loss > 0:
                ld = losses.detach()
                losses = losses / ld * torch.minimum(ld, clip_loss)

            term_loss = (losses * done).sum() / (done.sum() + 1e-20)
            flow_loss = (losses * (1 - done)).sum() / ((1 - done).sum() + 1e-20)
            if balanced_loss:
                loss = term_loss * leaf_coef + flow_loss
            else:
                loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()
            last_losses.append([loss.item()])
            # 5. Output log and evaluation results
            if (i + 1) % args.verbose == 0:
                # Logger
                train_infos.append((
                    exp_inflow.data.cpu().numpy(),
                    exp_outflow.data.cpu().numpy(),
                    reward.data.cpu().numpy(),
                    [i.pow(2).sum().item() for i in agent.parameters()]
                ))
                last_losses = np.mean(last_losses)
                # Policy network evaluation
                train_perf = self.eval_agent(agent, train_dataset, gnn_model)
                eval_perf = self.eval_agent(agent, eval_dataset, gnn_model)

                # Training info
                with open(f'log/gflowx-[{args.dataset}]-{datetime_now}.txt', 'a') as f:
                    f.write('Epoch %2d: lr=%.6f loss=%.3f, train_acc:%.3f, eval_acc:%.3f, time=%3.3f\n' % \
                            (i, lr, last_losses, train_perf['acc'], eval_perf['acc'], time.time() - time_last_check))

                print('Epoch %2d: lr=%.6f loss=%.3f, train_acc:%.3f, eval_acc:%.3f, time=%3.3f\n' % \
                 (i, lr, last_losses, train_perf['acc'], eval_perf['acc'], time.time() - time_last_check))
                time_last_check = time.time()
                last_losses = []

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(agent.parameters(),
                                                args.clip_grad)

            optimizer.step()
            scheduler.step()
            agent.training_steps = i + 1
            print(f"finish this epoch takes {time.time()-epoch_start}, {sampler.graph.idx}")


        sampler.stop_samplers_and_join()
        if exp_dir is not None:
            torch.save(agent, osp.join(exp_dir, 'agent.pt'))
        return agent

    def eval_agent(self, agent, dataset, gnn_model, ratio=0.8):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        acc_list = []
        for graph in loader:
            _, edge_imp = agent.foward_multisteps(graph, gnn_model, remove_ratio=1 - ratio)
            acc, _ = self.evaluate_acc(top_ratio_list=[0.1 * i for i in range(1, 11)],
                                       graph=graph, imp=edge_imp.float())
            acc_list.append(acc)
        return {'acc': np.mean(acc_list)}

    def _get_reward(self, full_graph_pred, subgraph_pred, target_y, pre_reward, mode='mutual_info'):
        '''Compute the rewards for the generated subgraphs'''

        if self.cf_flag:
            # for counterfactual explanations, we use the logits at the target class as rewards
            reward = torch.sum(target_y * torch.log(subgraph_pred + EPS), dim=1)
        else:
            # following RC-Explainer, we use three modes of rewards
            if mode == 'mutual_info':
                reward = torch.sum(full_graph_pred * torch.log(subgraph_pred + EPS), dim=1)
                reward += 2 * (target_y == subgraph_pred.argmax(dim=1)).float() - 1.
            elif mode == 'binary':
                reward = (target_y == subgraph_pred.argmax(dim=1)).float()
                reward = 2. * reward - 1.
            elif mode == 'cross_entropy':
                reward = torch.log(subgraph_pred + EPS)[:, target_y]

        reward += 0.97 * pre_reward
        return torch.sigmoid(reward)

    def explain_node_tasks(self):

        return None

