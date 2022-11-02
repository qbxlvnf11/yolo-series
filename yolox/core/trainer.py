#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger
import copy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from yolox.evaluators import eval

class Trainer:
    def __init__(self, exp, config_dict, device, dataset_name, class_list, class_num, resize_width_height):
        
        print('-'*60)
        
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.yolox_config = config_dict['yolox_config']
        self.train_config = config_dict['train_config']

        # training related attr
        self.max_epoch = self.train_config.getint('train', 'epochs')
        self.start_epoch = self.train_config.getint('train', 'start_epoch')
        self.fp16 = self.train_config.getboolean('data', 'fp16')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.device = device #"cuda:{}".format(self.local_rank)
        self.local_rank = int(str(self.device)[str(self.device).index(':')+1:]) #get_local_rank()
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt
        self.train_batch_size = self.train_config.getint('data', 'train_batch_size')
        self.eval_batch_size = self.train_config.getint('data', 'eval_batch_size')
        self.fine_tune_mode = self.train_config.getboolean('train', 'fine_tune_mode')
        self.only_backbone_weight_load_mode = self.train_config.getboolean('train', 'only_backbone_weight_load_mode')
        self.except_predict_layer_weight_load_mode = self.train_config.getboolean('train', 'except_predict_layer_weight_load_mode')
        self.cache_mode = self.train_config.getboolean('train', 'cache_mode')
        self.occupy = self.train_config.getboolean('train', 'occupy')
        self.logger = self.train_config.get('train', 'logger')
        self.print_interval = self.train_config.getint('train', 'print_interval')
        self.eval_interval = self.train_config.getint('train', 'eval_interval')
        print('Max epoch:', self.max_epoch)
        print('Distributed:', self.is_distributed)
        print('Start epoch:', self.start_epoch)
        print('Local rank:', self.local_rank)
        print('Train batch size:', self.train_batch_size)
        print('Evaluation batch size:', self.eval_batch_size)
        print('Fine tune mode:', self.fine_tune_mode)
        print('Only backbone weight load mode:', self.only_backbone_weight_load_mode)
        print('Except predict layer load mode:', self.except_predict_layer_weight_load_mode)
        print('Cache mode:', self.cache_mode)
	
        # Yolox model config
        self.weight_path = self.yolox_config.get('yolox', 'weight_path', raw=True)
        print('Weight path:', self.weight_path)
        
        # data/dataloader related attr
        self.dataset_name = dataset_name
        self.data_num_workers = self.train_config.getint('data', 'data_num_workers')
        self.data_type = torch.float16 if self.fp16 else torch.float32
        self.class_num = class_num
        self.resize_width_height = resize_width_height
        self.input_size = exp.img_size
        self.class_list = class_list
        self.best_ap = 0
        print('Data num workers:', self.data_num_workers)

        # Threshold
        self.object_confidence_threshold = self.yolox_config.getfloat('threshold', 'object_confidence_threshold')
        self.nms_thres = self.yolox_config.getfloat('threshold', 'nms_thres')
        self.val_iou_threshold = self.yolox_config.getfloat('threshold', 'val_iou_threshold')
        print('NMS threshold:', self.nms_thres)
        print('Conf threshold:', self.object_confidence_threshold)
        print('Val threshold:', self.val_iou_threshold)

        # metric record
        self.meter = MeterBuffer(window_size=self.print_interval)
        self.file_name = self.train_config.get('save', 'folder_path', raw=True)
        print('Metric record file folder:', self.file_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        '''setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )'''
        
        print('-'*60)

    def train(self):
        
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            print('Train error!', e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        self.optimizer.zero_grad()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.fp16):
            outputs = self.model(inps, targets)
        
        loss = outputs["total_loss"]
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        #logger.info("args: {}".format(self.args))
        #logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model().to(self.device)
        #logger.info("Model Summary: {}".format(get_model_info(model, (self.resize_width_height, self.resize_width_height))))
        
        # solver related init
        self.optimizer = self.exp.get_optimizer(self.train_batch_size)
	
        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)
        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.train_batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.cache_mode,
            data_num_workers=self.data_num_workers
        )
        
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.train_batch_size, self.max_iter
        )
        
        if self.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.eval_batch_size, is_distributed=self.is_distributed, data_num_workers=self.data_num_workers
        )
        
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")
            '''elif self.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
            )'''

        logger.info("Training start...")
        #logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        '''if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()'''

    def before_epoch(self):
        logger.info("---> start train epoch {}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")           
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")
          
    def after_epoch(self):
        #self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
            self.model.train()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            '''if self.rank == 0:
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)'''

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )
            #print('Random resizing:', self.input_size)

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        
        if self.fine_tune_mode:
            logger.info("fine-tune mode")
            if self.weight_path is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.weight_path
            ckpt = torch.load(ckpt_file, map_location=self.device)
            
            # resume the model/optimizer state dict
            load_layers_num = 0
            for key, param in ckpt["model"].items():
                
                if key not in model.state_dict():
                    continue
                    
                if self.only_backbone_weight_load_mode and key.find("head") == -1:
                    model.state_dict()[key].copy_(param)
                    load_layers_num += 1
                elif self.except_predict_layer_weight_load_mode and key.find("cls_preds") == -1:
                    model.state_dict()[key].copy_(param) 
                    load_layers_num += 1
                elif not self.only_backbone_weight_load_mode and not self.except_predict_layer_weight_load_mode:
                    model.state_dict()[key].copy_(param) 
                    load_layers_num += 1
            logger.info("Num of layers with pretrained weight: " + str(load_layers_num))
                    
            '''
            ckpt_model = {}
            if self.only_backbone_weight_load_mode:
                logger.info("Only load weights of backbone")
                for key in ckpt["model"].keys():
                    if key.find("head") == -1:
                    	c_key = key[9:] # Remove first 'backbone.'
                    	ckpt_model[c_key] = ckpt["model"][key]
                model.backbone.load_state_dict(ckpt_model)
                print(model.state_dict().keys())
            elif self.except_predict_layer_weight_load_mode:

            else:
                model.load_state_dict(ckpt["model"])
                '''
                
            #self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            
            # resume the training states variables
            start_epoch = (
                self.start_epoch - 1
                if self.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.fine_tune_mode, self.start_epoch
                )
            )  # noqa
        else:
            '''if self.weight_path is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.weight_path
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)'''
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        
        if self.dataset_name == 'coco2014' or self.dataset_name == 'coco2017':
            if self.use_model_ema:
                evalmodel = self.ema_model.ema
            else:
                evalmodel = self.model
                if is_parallel(evalmodel):
                    evalmodel = evalmodel.module

            with adjust_status(evalmodel, training=False):
                (ap50_95, ap50, summary), predictions = self.exp.eval(
                    evalmodel, self.evaluator, self.is_distributed, return_outputs=True
                )

            update_best_ckpt = ap50_95 > self.best_ap
            self.best_ap = max(self.best_ap, ap50_95)

            if self.rank == 0:
                if self.logger == "tensorboard":
                    self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                    self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
                '''if self.args.logger == "wandb":
                    self.wandb_logger.log_metrics({
                        "val/COCOAP50": ap50,
                        "val/COCOAP50_95": ap50_95,
                        "train/epoch": self.epoch + 1,
                    })
                    self.wandb_logger.log_images(predictions)'''
                logger.info("\n" + summary)
            synchronize()

            self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
            if self.save_history_ckpt:
                self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        else:
            loader = self.exp.get_eval_loader(batch_size=self.eval_batch_size, is_distributed=self.is_distributed, data_num_workers=self.data_num_workers, testdev=False, legacy=False)
            mAP = eval(self.device, self.model, loader, self.resize_width_height, self.class_list, class_num=self.class_num, nms_thres=self.nms_thres, object_confidence_threshold=self.object_confidence_threshold, val_iou_threshold=self.val_iou_threshold)

            if mAP is not None:
                update_best_ckpt = mAP > self.best_ap
                self.best_ap = max(self.best_ap, mAP)
                self.save_ckpt("last_epoch", update_best_ckpt, ap=mAP)
                if self.save_history_ckpt:
                    self.save_ckpt(f"epoch_{self.epoch + 1}", ap=mAP)
        print('Best ap:', self.best_ap)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}, {}".format(self.file_name, ckpt_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                #"optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            '''if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )'''
