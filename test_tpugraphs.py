import datetime
import os
import torch
import logging
import time
import copy
import torch.nn as nn
import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.data import Batch
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
import torch_geometric.nn as tnn
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import numpy as np
import scipy

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

class TPUModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.emb = nn.Embedding(128, 128, max_norm=True)
        # 256
        self.linear_map = nn.Linear(286, 128, bias=True)
        self.op_weights = nn.Parameter(torch.ones(1,1,requires_grad=True) * 100)
        self.config_weights = nn.Parameter(torch.ones(1,18,requires_grad=True) * 100)
        # get opcode, concat first, multiply config feature by 100 and learnable
        # no weights needed first
        # one first layer mlp
        # embedding dropout 
        # graph sage mean pooling 
        # no bias in gnn
        # drop 10% keep 90%
        # one mlp
        # l2 norm
        # 3 layer mlp 
        # no bias
        # now start graph embedding
        # concat of sum and max pooling
        # 1 layer mlp with bias, no activation
        
        # adam  with clip norm of value 1 start from 1e-5, later larger to 1e-3 max 

def pairwise_hinge_loss_batch(pred, true):
    # pred: (batch_size, num_preds )
    # true: (batch_size, num_preds)
    batch_size = pred.shape[0]
    num_preds = pred.shape[1]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = true[:,i_idx] > true[:,j_idx]
    loss = torch.sum(torch.nn.functional.relu(0.1 - (pred[:,i_idx] - pred[:,j_idx])) * pairwise_true.float()) / batch_size
    return loss

def preprocess_batch(batch, model, num_sample_configs):
    
    batch_list = batch.to_data_list()
    processed_batch_list = []
    for g in batch_list:
        sample_idx = torch.randint(0, g.num_config.item(), (num_sample_configs,))
        g.y = g.y[sample_idx]
        g.config_feats = g.config_feats.view(g.num_config, g.num_config_idx, -1)[sample_idx, ...]
        g.config_feats = g.config_feats.transpose(0,1)
        g.config_feats_full = torch.zeros((g.num_nodes, num_sample_configs, g.config_feats.shape[-1]), device=g.config_feats.device)
        g.config_feats_full[g.config_idx, ...] += g.config_feats
        g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(g.num_nodes, g.num_nodes))
        processed_batch_list.append(g)
    return Batch.from_data_list(processed_batch_list)

@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        num_sample_config = len(batch.y)
        batch = preprocess_batch(batch, model, num_sample_config)
        batch.split = split
        true = batch.y
        batch_list = batch.to_data_list()
        batch_seg_list = []
        batch_num_parts = []
        cnt = 0
        for i in range(len(batch_list)):
            num_parts = len(batch_list[i].partptr) - 1
            batch_num_parts.append(num_parts)
            for j in range(num_parts):
                start = int(batch_list[i].partptr.numpy()[j])
                length = int(batch_list[i].partptr.numpy()[j+1]) - start

                N, E = batch_list[i].num_nodes, batch_list[i].num_edges
                data = copy.copy(batch_list[i])
                del data.num_nodes
                adj, data.adj = data.adj, None

                adj = adj.narrow(0, start, length).narrow(1, start, length)
                edge_idx = adj.storage.value()

                for key, item in data:
                    if isinstance(item, torch.Tensor) and item.size(0) == N:
                        data[key] = item.narrow(0, start, length)
                    elif isinstance(item, torch.Tensor) and item.size(0) == E:
                        data[key] = item[edge_idx]
                    else:
                        data[key] = item

                row, col, _ = adj.coo()
                data.edge_index = torch.stack([row, col], dim=0)
                for k in range(len(data.y)):
                    unfold_g = Data(edge_index=data.edge_index, op_feats=data.op_feats, op_code=data.op_code, config_feats=data.config_feats_full[:, k, :], num_nodes=length)
                    if cnt % 32 == 0:
                        batch_seg_list.append([])
                    batch_seg_list[-1].append(unfold_g)
        res_list = []
        for batch_seg in batch_seg_list:
            batch_seg = Batch.from_data_list(batch_seg)
            batch_seg.to(torch.device(cfg.device))
            true = true.to(torch.device(cfg.device))
            # more preprocessing
            batch_seg.op_emb = model.emb(batch_seg.op_code.long())
            batch_seg.x = torch.cat((batch_seg.op_feats, batch_seg.op_emb * model.op_weights, batch_seg.config_feats * model.config_weights), dim=-1)
            batch_seg.x = model.linear_map(batch_seg.x)
        
            module_len = len(list(model.model.children()))
            for i, module in enumerate(model.model.children()):
                if i < module_len - 1:
                    batch_seg = module(batch_seg)
                if i == module_len - 1:
                    batch_seg_embed = tnn.global_max_pool(batch_seg.x, batch_seg.batch) + tnn.global_mean_pool(batch_seg.x, batch_seg.batch)
            graph_embed = batch_seg_embed / torch.norm(batch_seg_embed, dim=-1, keepdim=True)
            for i, module in enumerate(model.model.children()):
                if i == module_len - 1:
                    res = module.layer_post_mp(graph_embed)
                    res_list.append(res)
        res_list = torch.cat(res_list, dim=0)
        pred = torch.zeros(1, len(data.y), 1).to(torch.device(cfg.device))
        part_cnt = 0
        for i, num_parts in enumerate(batch_num_parts):
            for _ in range(num_parts):
                for j in range(num_sample_config):
                    pred[i, j, :] += res_list[part_cnt, :]
                    part_cnt += 1
        pred = pred.view(num_sample_config)
        true = true.view(num_sample_config)
        pred_rank = torch.argsort(pred, dim=-1, descending=False)
        true_rank = torch.argsort(true, dim=-1, descending=False)
        pred_rank = pred_rank.cpu().numpy()
        true_rank = true_rank.cpu().numpy()
        true = true.cpu().numpy()
        err_1 = (true[pred_rank[0]] - true[true_rank[0]]) / true[true_rank[0]]
        err_10 = (np.min(true[pred_rank[:10]]) - true[true_rank[0]]) / true[true_rank[0]]
        err_100 = (np.min(true[pred_rank[:100]]) - true[true_rank[0]]) / true[true_rank[0]]
        print('top 1 err: ' + str(err_1))
        print('top 10 err: ' + str(err_10))
        print('top 100 err: ' + str(err_100))
        print("kendall:" + str(scipy.stats.kendalltau(pred_rank, true_rank).correlation))
        time_start = time.time()

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        model = create_model()
        model = TPUModel(model)
        model = model.to(torch.device(cfg.device))
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        loaders = create_loader()
        loggers = create_logger()
        # Load checkpoint from run dir
        epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        eval_epoch(loggers[2], loaders[2], model, split='test')
        
        