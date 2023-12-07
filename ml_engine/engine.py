import datetime
import logging
import time

import mlflow
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ml_engine import utils
from ml_engine.lr_scheduler import build_scheduler
from ml_engine.optimizer import build_optimizer
from ml_engine.samplers import DistributedRepeatSampler, DistributedEvalSampler
from ml_engine.utils import configure_ddp, NativeScalerWithGradNormCount, AverageMeter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.local_rank, self.rank, self.world_size = configure_ddp()
        seed = self.cfg.seed + dist.get_rank()
        utils.set_seed(seed)
        cudnn.benchmark = True

        logger.info(f"Creating model:{self.cfg.model.type}/{self.cfg.model.name}")
        model = self.build_model(self.cfg.model)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")

        if self.cfg.model.pretrained:
            state_dict = self.get_state_dict(self.cfg.model.pretrained)
            model.load_state_dict(state_dict)

        model.cuda()
        model_wo_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], broadcast_buffers=False)

        self.min_loss = 99999
        self.model = model
        self.model_wo_ddp = model_wo_ddp

        self.data_loader_registers = {}

    def get_state_dict(self, model_path):
        state_dict_uri = mlflow.get_artifact_uri(model_path)
        return mlflow.pytorch.load_state_dict(state_dict_uri)

    def resume_state_dict(self, module, artifact_path):
        state_dict_uri = mlflow.get_artifact_uri(artifact_path)
        state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
        module.load_state_dict(state_dict)
        logger.info(f'State dict {artifact_path} is loaded')

    def save_state_dict(self, module, artifact_path):
        state_dict = module.state_dict()
        mlflow.pytorch.log_state_dict(state_dict, artifact_path)
        logger.info(f'State dict {artifact_path} is saved')

    def build_model(self, model_conf):
        raise NotImplementedError()

    def get_transforms(self):
        raise NotImplementedError()

    def load_dataset(self, mode, data_conf):
        raise NotImplementedError()

    def get_dataloader(self, mode, dataset, repeat):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        num_tasks = self.world_size
        global_rank = self.rank
        if mode == 'train':
            sampler = DistributedRepeatSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, repeat=repeat)

            data_loader = DataLoader(
                dataset, sampler=sampler,
                batch_size=self.cfg.data.batch_size,
                num_workers=self.cfg.data.num_workers,
                pin_memory=self.cfg.data.pin_memory,
                drop_last=True,
            )
        else:
            sampler = DistributedEvalSampler(
                dataset, shuffle=self.cfg.test.shuffle, rank=global_rank, num_replicas=num_tasks, repeat=repeat)

            data_loader = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=self.cfg.data.test_batch_size,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=self.cfg.data.pin_memory,
                drop_last=False
            )
        self.data_loader_registers[mode] = data_loader
        return data_loader

    def train(self, repeat_train_data=1, ref_lr_bs=256.):
        mode = 'train'
        dataset = self.load_dataset(mode, self.cfg.data)
        data_loader = self.get_dataloader(mode, dataset, repeat_train_data)

        # linear scale the learning rate according to total batch size, may not be optimal
        batch_size = self.cfg.data.batch_size * dist.get_world_size()
        self.cfg.train.base_lr = self.cfg.train.base_lr * batch_size / ref_lr_bs
        self.cfg.lr_scheduler.warmup_lr = self.cfg.lr_scheduler.warmup_lr * batch_size / ref_lr_bs
        self.cfg.lr_scheduler.min_lr = self.cfg.lr_scheduler.min_lr * batch_size / ref_lr_bs

        # gradient accumulation also need to scale the learning rate
        if self.cfg.train.accumulation_steps > 1:
            self.cfg.train.base_lr = self.cfg.train.base_lr * self.cfg.train.accumulation_steps
            self.cfg.lr_scheduler.warmup_lr = self.cfg.lr_scheduler.warmup_lr * self.cfg.train.accumulation_steps
            self.cfg.lr_scheduler.min_lr = self.cfg.lr_scheduler.min_lr * self.cfg.train.accumulation_steps

        optimizer = build_optimizer(self.cfg.train, self.cfg.optimizer, self.model_wo_ddp)
        loss_scaler = NativeScalerWithGradNormCount()
        lr_scheduler = build_scheduler(self.cfg.lr_scheduler, self.cfg.train.epochs, optimizer,
                                       len(data_loader) // self.cfg.train.accumulation_steps)

        criterion = self.get_criterion()

        if self.cfg.train.auto_resume:
            self.resume_state_dict(self.model_wo_ddp, 'models:/model/latest')
            self.resume_state_dict(optimizer, 'models:/optimizer/latest')
            self.resume_state_dict(lr_scheduler, 'models:/lr_scheduler/latest')
            self.resume_state_dict(loss_scaler, 'models:/lost_scaler/latest')

        logger.info("Start training...")
        start_time = time.time()
        for epoch in range(self.cfg.TRAIN.START_EPOCH, self.cfg.TRAIN.EPOCHS):
            self.train_one_epoch(epoch, data_loader, optimizer, lr_scheduler, loss_scaler, criterion)

            if dist.get_rank() == 0 and (epoch % self.cfg.save_freq == 0 or epoch == (self.cfg.train.epochs - 1)):
                self.save_state_dict(self.model_wo_ddp, 'models:/model/latest')
                self.save_state_dict(optimizer, 'models:/optimizer/latest')
                self.save_state_dict(lr_scheduler, 'models:/lr_scheduler/latest')
                self.save_state_dict(loss_scaler, 'models:/lost_scaler/latest')

            loss = self.validate()
            if loss < self.min_loss:
                self.save_state_dict(self.model_wo_ddp, 'models:/model/best')

            self.min_loss = min(self.min_loss, loss)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

    def train_step(self, samples):
        return self.model(samples)

    def prepare_data(self, samples, targets):
        return samples, targets

    def train_one_epoch(self, epoch, data_loader, optimizer, lr_scheduler, loss_scaler, criterion):
        self.model.train()
        optimizer.zero_grad()
        data_loader.sampler.set_epoch(epoch)
        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (samples, targets) in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            samples, targets = self.prepare_data(samples, targets)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_enable):
                outputs = self.train_step(samples)

            loss = criterion(outputs, targets)
            loss = loss / self.cfg.TRAIN.ACCUMULATION_STEPS

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=self.cfg.train.clip_grad,
                                    parameters=self.model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % self.cfg.train.accumulation_steps == 0)

            if (idx + 1) % self.cfg.train.accumulation_steps == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // self.cfg.train.accumulation_steps)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item() * self.cfg.train.accumulation_steps, targets.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)

            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.cfg.print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{self.cfg.train.epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        loss_meter.all_reduce()
        return loss_meter.avg

    def get_criterion(self):
        raise NotImplementedError()

    @torch.no_grad()
    def validate(self):
        raise NotImplementedError()

    def throughput(self):
        self.model.eval()
        mode = 'validation'
        dataset = self.load_dataset(mode, self.cfg.data)
        data_loader = self.get_dataloader('validation', dataset, repeat=1)
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                self.model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            tic1 = time.time()
            for i in range(30):
                self.model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            throughput_val = 30 * batch_size / (tic2 - tic1)
            logger.info(f"batch_size {batch_size} throughput {throughput_val}")
            return throughput_val
