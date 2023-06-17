import logging
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tnn
from torch_geometric.data import Batch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
from graphgps.history import History

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


def pairwise_hinge_loss(pred, true):
    num_preds = pred.shape[0]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = true[i_idx] > true[j_idx]
    loss = torch.sum(torch.nn.functional.relu(0.1 - (pred[i_idx] - pred[j_idx])) * pairwise_true.float())
    opa_indices = pairwise_true.nonzero().flatten()
    opa_preds = pred[i_idx[opa_indices]] - pred[j_idx[opa_indices]]
    return loss, pairwise_true[opa_indices], opa_preds

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
    return Batch.from_data_list(processed_batch_list), sample_idx
        
def train_epoch(logger, loader, model, optimizer, scheduler, emb_table, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    num_sample_config = 32
    for iter, batch in enumerate(loader):
        batch, sampled_idx = preprocess_batch(batch, model, num_sample_config)
        batch.to(torch.device(cfg.device))
        true = batch.y
        
        batch_list = batch.to_data_list()
        batch_train_list = []
        batch_other = []
        batch_num_parts = []
        segments_to_train = []
        for i in range(len(batch_list)):
            num_parts = len(batch_list[i].partptr) - 1
            batch_num_parts.extend([num_parts] * num_sample_config)
            segment_to_train = np.random.randint(num_parts)
            segments_to_train.append(segment_to_train)
            for j in range(num_parts):
                start = int(batch_list[i].partptr.cpu().numpy()[j])
                length = int(batch_list[i].partptr.cpu().numpy()[j+1]) - start

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
                if j == segment_to_train:
                    for k in range(len(data.y)):
                        unfold_g = Data(edge_index=data.edge_index, op_feats=data.op_feats, op_code=data.op_code, config_feats=data.config_feats_full[:, k, :], num_nodes=length)
                        batch_train_list.append(unfold_g)
                else:
                    for k in range(len(data.y)):
                        batch_other.append(emb_table.pull(batch_list[i].partition_idx.cpu()+num_parts*sampled_idx[k]+j))       
        batch_train = Batch.from_data_list(batch_train_list)
        batch_train = batch_train.to(torch.device(cfg.device))
        true = true.to(torch.device(cfg.device))
        # more preprocessing
        batch_train.split = 'train'
        batch_train.op_emb = model.emb(batch_train.op_code.long())
        batch_train.x = torch.cat((batch_train.op_feats, batch_train.op_emb * model.op_weights, batch_train.config_feats * model.config_weights), dim=-1)
        batch_train.x = model.linear_map(batch_train.x)
        
        module_len = len(list(model.model.children()))
        for i, module in enumerate(model.model.children()):
            if i < module_len - 1:
                batch_train = module(batch_train)
            if i == module_len - 1:
                batch_train_embed = tnn.global_max_pool(batch_train.x, batch_train.batch) + tnn.global_mean_pool(batch_train.x, batch_train.batch)
        graph_embed = batch_train_embed / torch.norm(batch_train_embed, dim=-1, keepdim=True)
        for i, module in enumerate(model.model.children()):
            if i == module_len - 1:
                graph_embed = module.layer_post_mp(graph_embed)
        
        binomial = torch.distributions.binomial.Binomial(probs=0.5)
        if len(batch_other) > 0:
            batch_other = torch.cat(batch_other, dim=0)
            mask =  binomial.sample((batch_other.shape[0], 1)).to(torch.device(cfg.device))
            batch_other = batch_other.to(torch.device(cfg.device))
            batch_other = batch_other * mask    
            batch_other_embed = torch.zeros_like(graph_embed)
            part_cnt = 0
            for i, num_parts in enumerate(batch_num_parts):
                for j in range(num_parts-1):
                    batch_other_embed[i, :] += batch_other[part_cnt, :]
                    part_cnt += 1
            batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
            batch_num_parts = batch_num_parts.view(-1, 1)
            multiplier_num = (batch_num_parts-1)/ 2 + 1
            pred = graph_embed*multiplier_num + batch_other_embed
        else:
            pred = graph_embed
        
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'TPUGraphs':
            pred = pred.view(-1, num_sample_config)
            true = true.view(-1, num_sample_config)
            loss = pairwise_hinge_loss_batch(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        for i in range(graph_embed.shape[0]):
            push_idx = batch_list[i // num_sample_config].partition_idx.cpu()+sampled_idx[i % num_sample_config]*(len(batch_list[i // num_sample_config].partptr)-1)+segments_to_train[i // num_sample_config]
            emb_table.push(graph_embed[i].cpu(), push_idx)
        
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    num_sample_config = 32
    for batch in loader:
        batch, _ = preprocess_batch(batch, model, num_sample_config)
        batch.split = split
        true = batch.y
        batch_list = batch.to_data_list()
        batch_seg = []
        batch_num_parts = []
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
                    batch_seg.append(unfold_g)
        batch_seg = Batch.from_data_list(batch_seg)
        batch_seg.to(torch.device(cfg.device))
        true = true.to(torch.device(cfg.device))
        # more preprocessing
        # batch_train.config_feats = model.config_feats_transform(batch_train.config_feats)
        batch_seg.op_emb = model.emb(batch_seg.op_code.long())
        # batch_train.op_feats = model.op_feats_transform(batch_train.op_feats)
        batch_seg.x = torch.cat((batch_seg.op_feats, model.op_weights * batch_seg.op_emb, batch_seg.config_feats * model.config_weights), dim=-1)
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
        pred = torch.zeros(len(loader.dataset), len(data.y), 1).to(torch.device(cfg.device))
        part_cnt = 0
        for i, num_parts in enumerate(batch_num_parts):
            for _ in range(num_parts):
                for j in range(num_sample_config):
                    pred[i, j, :] += res[part_cnt, :]
                    part_cnt += 1
        batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
        batch_num_parts = batch_num_parts.view(-1, 1)
        extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'TPUGraphs':
            pred = pred.view(-1, num_sample_config)
            true = true.view(-1, num_sample_config)
            loss = pairwise_hinge_loss_batch(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()


@register_train('custom_tpu')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    model = model.to(cfg.device)
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    emb_table = History(500000000, 1)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, emb_table,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-tpu')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))
    val_perf = perf[1]

    best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                             cfg.metric_agg)()
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()
