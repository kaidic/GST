import logging
import time
import copy
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.optim import create_optimizer, OptimizerConfig
    
def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
from graphgps.history import History


def train_epoch(logger, loader, model, optimizer, scheduler, emb_table, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    module_len = len(list(model.children()))
    for iter, batch in enumerate(loader):
        true = batch.y
        num_sample = len(batch.y)
        emb = emb_table.pull(torch.tensor(0))
        batch_list = batch.to_data_list()
        graph_embed = torch.zeros(num_sample, emb.shape[1])
        batch_num_parts = []
        for i in range(num_sample):
            num_parts = len(batch_list[i].partptr) - 1
            batch_num_parts.append(num_parts)
        for i, num_parts in enumerate(batch_num_parts):
            for j in range(num_parts):
                graph_embed[i, :] += emb_table.pull(torch.tensor(batch[i].partition_idx+j)).clone().detach().flatten()
        graph_embed = graph_embed.to(torch.device(cfg.device))
        batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
        batch_num_parts = batch_num_parts.view(-1, 1)
        graph_embed = graph_embed / batch_num_parts
        true = true.to(torch.device(cfg.device))
        for i, module in enumerate(model.children()):
            if i == module_len - 1:
                pred = module.layer_post_mp(graph_embed)
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
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
    for batch in loader:
        batch.split = split
        true = batch.y
        batch_list = batch.to_data_list()
        batch_seg_list = []
        batch_num_parts = []
        cnt = 0
        for i in range(len(batch.y)):
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
                if cnt % 32 == 0:
                    batch_seg_list.append([])
                batch_seg_list[-1].append(data)
                cnt += 1
        batch_seg_embed_list = []
        true = true.to(torch.device(cfg.device))
        for batch_seg in batch_seg_list:
            batch_seg = Batch.from_data_list(batch_seg)
            batch_seg.to(torch.device(cfg.device))
            module_len = len(list(model.children()))
            for i, module in enumerate(model.children()):
                if i < module_len - 1:
                    batch_seg = module(batch_seg)
                if i == module_len - 1:
                    batch_seg_embed = module.pooling_fun(batch_seg.x, batch_seg.batch)
                    batch_seg_embed_list.append(batch_seg_embed)
        batch_seg_embed_list = torch.cat(batch_seg_embed_list, dim=0)
        graph_embed = torch.zeros(len(batch.y), batch_seg_embed.shape[1]).to(torch.device(cfg.device))
        part_cnt = 0
        for i, num_parts in enumerate(batch_num_parts):
            for j in range(num_parts):
                graph_embed[i, :] += batch_seg_embed_list[part_cnt, :]
                part_cnt += 1
        batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
        batch_num_parts = batch_num_parts.view(-1, 1)
        graph_embed = graph_embed / batch_num_parts
        for i, module in enumerate(model.children()):
            if i == module_len - 1:
                pred = module.layer_post_mp(graph_embed)
       
        extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
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


@register_train('custom_finetune')
def custom_finetune(loggers, loaders, model, optimizer, scheduler):
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
    cfg.run_dir = 'tests/results/malnettiny-gps-etf-ds-sage/0'
    start_epoch = load_ckpt(model, optimizer, scheduler,
                            cfg.train.epoch_resume)
    logging.info('Start from epoch %s', start_epoch)
    optimizer = create_optimizer(model.post_mp.parameters(),
                                     new_optimizer_config(cfg))
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
    emb_table = History(60000, 300)
    inference_emb_table(loggers[0], loaders[0], model, emb_table)
    for cur_epoch in range(0, cfg.optim.max_epoch):
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

@torch.no_grad()
def inference_emb_table(logger, loader, model, emb_table):
    model.eval()
    time_start = time.time()
    for batch in loader:
        true = batch.y
        batch_list = batch.to_data_list()
        batch_seg_list = []
        batch_num_parts = []
        cnt = 0
        for i in range(len(batch.y)):
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
                if cnt % 32 == 0:
                    batch_seg_list.append([])
                batch_seg_list[-1].append(data)
                cnt += 1
        batch_seg_embed_list = []
        true = true.to(torch.device(cfg.device))
        for batch_seg in batch_seg_list:
            batch_seg = Batch.from_data_list(batch_seg)
            batch_seg.to(torch.device(cfg.device))
            module_len = len(list(model.children()))
            for i, module in enumerate(model.children()):
                if i < module_len - 1:
                    batch_seg = module(batch_seg)
                if i == module_len - 1:
                    batch_seg_embed = module.pooling_fun(batch_seg.x, batch_seg.batch)
                    batch_seg_embed_list.append(batch_seg_embed)
        batch_seg_embed_list = torch.cat(batch_seg_embed_list, dim=0)
        part_cnt = 0
        for i, num_parts in enumerate(batch_num_parts):
            for j in range(num_parts):
                emb_table.push(batch_seg_embed_list[part_cnt].cpu(), batch[i].partition_idx+j)
                part_cnt += 1