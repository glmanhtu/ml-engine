import datetime
import os
import time
from functools import lru_cache
from typing import Dict

import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ml_engine import utils
from ml_engine.data.samplers import DistributedRepeatableEvalSampler
from ml_engine.evaluation.metrics import AverageMeter
from ml_engine.logger import create_logger
from ml_engine.lr_scheduler import build_scheduler
from ml_engine.modelling.resnet import ResNet32MixConv
from ml_engine.modelling.simsiam import SimSiamV2CE, SimSiamV2
from ml_engine.optimizer import build_optimizer
from ml_engine.tracking.tracker import Tracker
from ml_engine.utils import get_ddp_config, NativeScalerWithGradNormCount, extract_params_from_omegaconf_dict, \
    revert_sync_batchnorm


class Trainer:
    def __init__(self, cfg: DictConfig, tracker: Tracker):
        self._cfg = cfg
        self._tracker = tracker
        self.local_rank, self.rank, self.world_size = get_ddp_config()
        exp_log_dir = os.path.join(cfg.log_dir, cfg.run.name)
        os.makedirs(exp_log_dir, exist_ok=True)
        self.logger = create_logger(exp_log_dir, self.rank, cfg.run.name, cfg.mode)
        self.logger.info(f"RANK and WORLD_SIZE in environ: {self.rank}/{self.world_size}")

        seed = self._cfg.seed + self.rank
        utils.set_seed(seed)
        cudnn.benchmark = True

        self.logger.info(f"Creating model {self._cfg.model.type}")
        model = self.build_model(self._cfg.model)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of params: {n_parameters}")

        if self._cfg.model.weights:
            state_dict = self._tracker.get_state_dict(self._cfg.model.weights)
            state_dict = self.prepare_pretrained_model(self._cfg.model.type, model.state_dict(), state_dict)
            model.load_state_dict(state_dict)

        model.cuda()
        model_wo_ddp = model
        model = DistributedDataParallel(model, device_ids=[self.local_rank], broadcast_buffers=False)

        self._min_loss = 99999
        self._model = model
        self._model_wo_ddp = model_wo_ddp
        self.__step = 0

    def prepare_pretrained_model(self, model_type, model_state_dict, pretrained_state_dict):
        """
        Modify the pretrained state dict before loading the model
        @param model_type:
        @param model_state_dict:
        @param pretrained_state_dict:
        @return: modified pretrained state_dict
        """
        if 'head.bias' in pretrained_state_dict:
            head_bias_pretrained = pretrained_state_dict['head.bias']
            n_c1 = head_bias_pretrained.shape[0]
            n_c2 = model_state_dict['head.bias'].shape[0]
            if n_c1 != n_c2:
                pretrained_state_dict['head.weight'] = model_state_dict['head.weight']
                pretrained_state_dict['head.bias'] = model_state_dict['head.bias']
        return pretrained_state_dict

    def resume_state_dict(self, module, artifact_path):
        state_dict = self._tracker.get_state_dict(artifact_path)
        module.load_state_dict(state_dict)
        self.logger.info(f'State dict {artifact_path} is loaded')

    def build_model(self, model_conf):
        if model_conf.type == 'ss2ce':
            model = SimSiamV2CE(
                arch=model_conf.arch,
                pretrained=model_conf.pretrained,
                dim=model_conf.embed_dim,
                pred_dim=model_conf.pred_dim,
                dropout=model_conf.dropout,
                n_classes=model_conf.n_classes)

        elif model_conf.type == 'ss2':
            model = SimSiamV2(
                arch=model_conf.arch,
                pretrained=model_conf.pretrained,
                dim=model_conf.embed_dim,
                pred_dim=model_conf.pred_dim,
                dropout=model_conf.dropout)

        elif model_conf.type == 'mixconv':
            model = ResNet32MixConv(
                img_size=(self._cfg.data.img_size, self._cfg.data.img_size),
                backbone=model_conf.arch,
                out_channels=model_conf.out_channels,
                mix_depth=model_conf.mix_depth,
                out_rows=model_conf.out_rows,
                weights=model_conf.pretrained,
                layers_to_freeze=model_conf.layers_freeze)

        else:
            timm_models = timm.list_models(pretrained=model_conf.pretrained)
            if model_conf.arch in timm_models:
                return timm.create_model(model_conf.arch, pretrained=model_conf.pretrained,
                                         num_classes=model_conf.num_classes)
            raise NotImplementedError(f'Network {model_conf.type} is not implemented!')

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def get_transform(self, mode, data_conf):
        raise NotImplementedError()

    def load_dataset(self, mode, data_conf, transform):
        raise NotImplementedError()

    @lru_cache
    def get_dataloader(self, mode, dataset, data_conf):
        num_tasks = self.world_size
        global_rank = self.rank
        sampler = DistributedRepeatableEvalSampler(dataset, shuffle=False, rank=global_rank, num_replicas=num_tasks)
        if mode == 'train':
            sampler = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            data_loader = DataLoader(
                dataset, sampler=sampler,
                batch_size=data_conf.batch_size,
                num_workers=data_conf.num_workers,
                pin_memory=data_conf.pin_memory,
                drop_last=True,
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=data_conf.test_batch_size,
                shuffle=False,
                num_workers=data_conf.num_workers,
                pin_memory=data_conf.pin_memory,
                drop_last=False
            )
        return data_loader

    def train(self, ref_lr_bs=256., mode='train'):
        if self._cfg.train.resume:
            self._cfg.train.start_epoch = int(self._tracker.get_metric('epoch')[-1].value) + 1
            self.__step = self._tracker.get_metric('train_loss')[-1].step
            self._min_loss = min([x.value for x in self._tracker.get_metric('val_loss')])
        else:
            self._tracker.log_params(extract_params_from_omegaconf_dict(self._cfg))
        dataset = self.load_dataset(mode, self._cfg.data, self.get_transform(mode, self._cfg.data))
        self.logger.info(f"Training data size: {len(dataset)}")
        data_loader = self.get_dataloader(mode, dataset, self._cfg.data)

        # linear scale the learning rate according to total batch size, may not be optimal
        batch_size = self._cfg.data.batch_size * dist.get_world_size()
        self._cfg.train.base_lr = self._cfg.train.base_lr * batch_size / ref_lr_bs
        self._cfg.lr_scheduler.warmup_lr = self._cfg.lr_scheduler.warmup_lr * batch_size / ref_lr_bs
        self._cfg.lr_scheduler.min_lr = self._cfg.lr_scheduler.min_lr * batch_size / ref_lr_bs

        # gradient accumulation also need to scale the learning rate
        if self._cfg.train.accumulation_steps > 1:
            self._cfg.train.base_lr = self._cfg.train.base_lr * self._cfg.train.accumulation_steps
            self._cfg.lr_scheduler.warmup_lr = self._cfg.lr_scheduler.warmup_lr * self._cfg.train.accumulation_steps
            self._cfg.lr_scheduler.min_lr = self._cfg.lr_scheduler.min_lr * self._cfg.train.accumulation_steps

        optimizer = build_optimizer(self._cfg.train, self._cfg.optimizer, self._model_wo_ddp)
        loss_scaler = NativeScalerWithGradNormCount()
        lr_scheduler = build_scheduler(self._cfg.lr_scheduler, self._cfg.train.epochs, optimizer,
                                       len(data_loader) // self._cfg.train.accumulation_steps)

        criterion = self.get_criterion()

        if self._cfg.train.resume:
            self.resume_state_dict(self._model_wo_ddp, 'models/model/latest')
            self.resume_state_dict(optimizer, 'models/optimizer/latest')
            self.resume_state_dict(lr_scheduler, 'models/lr_scheduler/latest')
            self.resume_state_dict(loss_scaler, 'models/lost_scaler/latest')

        loss = self.validate()
        self.log_metrics({'init_loss': loss})

        self.logger.info("Start training...")
        start_time = time.time()
        for epoch in range(self._cfg.train.start_epoch, self._cfg.train.epochs):

            if loss < self._min_loss:
                self.log_metrics({'best_loss': loss})
                self.logger.info(f"Init loss: {loss}")

            self.train_one_epoch(epoch, data_loader, optimizer, lr_scheduler, loss_scaler, criterion)

            if self.rank == 0 and (epoch % self._cfg.save_freq == 0 or epoch == (self._cfg.train.epochs - 1)):
                self._tracker.log_state_dict(self._model_wo_ddp.state_dict(), 'models/model/latest')
                self._tracker.log_state_dict(optimizer.state_dict(), 'models/optimizer/latest')
                self._tracker.log_state_dict(lr_scheduler.state_dict(), 'models/lr_scheduler/latest')
                self._tracker.log_state_dict(loss_scaler.state_dict(), 'models/lost_scaler/latest')

            loss = self.validate()
            self.log_metrics({'val_loss': loss})

            if loss < self._min_loss:
                self._tracker.log_state_dict(self._model_wo_ddp.state_dict(), 'models/model/best')

                self.log_metrics({'best_loss': loss})
                self.logger.info(f"Loss is reduced from {self._min_loss} to {loss}")

            self._min_loss = min(self._min_loss, loss)
            self.log_metrics({'epoch': epoch})

        self.resume_state_dict(self._model_wo_ddp, 'models/model/best')
        samples, _ = next(iter(data_loader))
        self._log_model(self._model_wo_ddp, samples)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Training time {}'.format(total_time_str))

    def _log_model(self, model, samples):
        reverted_bn_model = revert_sync_batchnorm(model)
        reverted_bn_model.eval()
        scripted_model = torch.jit.script(reverted_bn_model)
        self._tracker.log_model(scripted_model, 'models/model/final')

    def train_step(self, samples):
        self.__step += 1
        return self._model(samples)

    def prepare_data(self, samples, targets):
        return samples, targets

    def train_one_epoch(self, epoch, data_loader, optimizer, lr_scheduler, loss_scaler, criterion):
        self._model.train()
        optimizer.zero_grad()
        data_loader.sampler.set_epoch(epoch)
        batch_time = AverageMeter()
        loss_meter, norm_meter, scaler_meter = AverageMeter(), AverageMeter(), AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (samples, targets) in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            samples, targets = self.prepare_data(samples, targets)
            with torch.cuda.amp.autocast(enabled=self._cfg.amp_enable):
                outputs = self.train_step(samples)

            loss = criterion(outputs, targets)
            loss = loss / self._cfg.train.accumulation_steps

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=self._cfg.train.clip_grad,
                                    parameters=self._model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % self._cfg.train.accumulation_steps == 0)

            if (idx + 1) % self._cfg.train.accumulation_steps == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * len(data_loader) + idx) // self._cfg.train.accumulation_steps)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item() * self._cfg.train.accumulation_steps, targets.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)

            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self._cfg.print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (len(data_loader) - idx)
                self.logger.info(
                    f'Train: [{epoch}/{self._cfg.train.epochs}][{idx}/{len(data_loader)}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

                self.log_metrics({'train_loss': loss_meter.val})

        epoch_time = time.time() - start
        self.logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        loss_meter.all_reduce()
        return loss_meter.avg

    def log_metrics(self, metrics: Dict[str, float]):
        self._tracker.log_metrics(metrics, self.__step)

    def get_criterion(self):
        raise NotImplementedError()

    def validate_one_epoch(self, data_loader):
        raise NotImplementedError()

    @torch.no_grad()
    def validate(self, mode='validation'):
        self._model.eval()
        dataset = self.load_dataset(mode, self._cfg.data, self.get_transform(mode, self._cfg.data))
        self.logger.info(f"Validating data size: {len(dataset)}")

        data_loader = self.get_dataloader(mode, dataset, self._cfg.data)
        return self.validate_one_epoch(data_loader)

    def throughput(self, mode='validation'):
        self._model.eval()
        dataset = self.load_dataset(mode, self._cfg.data, self.get_transform(mode, self._cfg.data))
        data_loader = self.get_dataloader(mode, dataset, self._cfg.data)
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                self._model(images)
            torch.cuda.synchronize()
            self.logger.info(f"throughput averaged with 30 times")
            tic1 = time.time()
            for i in range(30):
                self._model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            throughput_val = 30 * batch_size / (tic2 - tic1)
            self.logger.info(f"batch_size {batch_size} throughput {throughput_val}")
            return throughput_val

    def __del__(self):
        dist.destroy_process_group()