
import os
from argparse import Namespace
import torch.types
from models.ddim_bitdit_class import BitDit
from data.pos.pos_dataset import LabelSet, POSDataset, Collator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
import wandb
from tqdm import tqdm
from prettytable import PrettyTable


class Trainer:
    def __init__(self, args: Namespace):
        self.args = args
        self._print_hyperparameters()
        if self.args.logger == 'wandb':
            # init logger
            run_name = "--".join([str(args.lr_bert), str(args.lr_other), str(args.max_epochs)])
            wandb.init(project="DiffusionPOS", name=run_name)
            wandb.config.update(self.args)
            wandb.define_metric("f1", summary="max")
        self.device = self._configure_device()
        self.dataset_path = os.path.join(os.getcwd(), 'datasets', self.args.dataset)
        self.label_set = LabelSet(self.args.dataset)
        if self.args.num_classes != len(self.label_set):
            print(
                f"the number of classes({self.args.num_classes}) you input is not equal from the statistic of dataset({len(self.label_set)})")
            print(f"automatically set num_classes to {len(self.label_set)} from {self.args.num_classes}")
            self.args.num_classes = len(self.label_set)

        self.model = BitDit(device=self.device,
                            num_classes=self.args.num_classes,
                            backbone=self.args.backbone,
                            time_steps=self.args.time_steps,
                            sampling_steps=self.args.sampling_steps,
                            noise_schedule=self.args.noise_schedule,
                            ddim_sampling_eta=self.args.ddim_sampling_eta,
                            self_condition=self.args.self_condition,
                            snr_scale=self.args.snr_scale,
                            dataset=self.args.dataset,
                            dim_model=self.args.dim_model,
                            dim_time=self.args.dim_time,
                            objective=self.args.objective,
                            loss_type=self.args.loss_type,
                            add_lstm=self.args.add_lstm,
                            freeze_bert=self.args.freeze_bert,
                            max_length=self.args.max_length,
                            depth=self.args.depth, 
                            num_labels=len(self.label_set))
        if self.args.logger == "wandb":
            wandb.watch(self.model, log_freq=1000)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
        self.collate_fn = Collator(self.tokenizer, self.args.max_length)

        self.train_dataloader = self._get_dataloader('train', self.args.batch_size)
        self.dev_dataloader = self._get_dataloader('dev', self.args.batch_size)
        self.test_dataloader = self._get_dataloader('test', self.args.batch_size)
        self.steps = self.args.max_steps

        self.optimizer, self.lr_scheduler = \
            self._configure_optimizer_and_scheduler(self.args.optimizer_type, self.args.lr_scheduler_type)

    def _get_dataloader(self, mode: str, bsz: int):
        assert mode in ['train', 'dev', 'test']
        dataset = POSDataset(self.args.dataset, mode, self.label_set)
        dataloader = DataLoader(dataset,
                                batch_size=bsz,
                                num_workers=self.args.num_workers,
                                drop_last=False,
                                shuffle=True if mode == "train" else False,
                                collate_fn=self.collate_fn)
        return dataloader

    def _print_hyperparameters(self):
        hparams = PrettyTable()
        hparams.title = 'Hyper Parameters'
        hparams.field_names = ["Name", "Value"]
        hparams.add_rows([[k, v] for k, v in self.args.__dict__.items()])
        print(hparams)

    def _configure_device(self):
        device = 'cpu'
        if self.args.use_gpu and torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} gpus are available!")
            device = torch.device(self.args.gpus)
            print(f"current gpu information:")
            cuda_property = torch.cuda.get_device_properties(device)
            print(f"number: {device}\t\tname: {cuda_property.name}\t\tmemory: {cuda_property.total_memory}")
        else:
            print("gpu is not available!")

        return device

    def _configure_optimizer_and_scheduler(self, optimizer_type: str, lr_scheduler_type: str):
        assert optimizer_type in ['AdamW'], f'do not support {optimizer_type}'
        assert lr_scheduler_type in ['linear', 'cosine', 'constant', 'cosine_hard_restart'], \
            f'do not support {lr_scheduler_type}'
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in self.model.named_parameters() if
                        not any(nd in n for nd in no_decay) and 'backbone' in n],
             'weight_decay': self.args.weight_decay,
             'lr': self.args.lr_bert},
            {'params': [p for n, p in self.model.named_parameters() if
                        any(nd in n for nd in no_decay) and 'backbone' in n],
             'weight_decay': 0.0,
             'lr': self.args.lr_bert},
            {'params': [p for n, p in self.model.named_parameters() if 'backbone' not in n],
             'weight_decay': self.args.weight_decay,
             'lr': self.args.lr_other},
        ]
        max_lrs = [self.args.lr_bert, self.args.lr_bert, self.args.lr_other]
        if self.args.freeze_bert:
            optimizer_params = optimizer_params[2:]
            max_lrs = [self.args.lr_other]
        optimizer = AdamW(optimizer_params)
        total_steps = self.args.max_epochs * len(self.train_dataloader)
        # num_warmup_steps = 0
        # if not self.args.warmup_steps:
        #     num_warmup_steps = self.args.warmup_steps
        # if not self.args.warmup_ratio:
        #     num_warmup_steps = total_steps * self.args.warmup_ratio
        # scheduler = get_lr_scheduler(name=lr_scheduler_type,
        #                              optimizer=optimizer,
        #                              num_warmup_steps=num_warmup_steps,
        #                              num_training_steps=total_steps,
        #                              num_cycles=self.args.num_cycles)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=max_lrs, total_steps=total_steps)
        return optimizer, scheduler

    def _step(self, batch):
        input_ids, attention_mask, seq_labels = [x.to(self.device) for x in batch]
        model_outputs = self.model(input_ids, attention_mask, seq_labels, ensemble=self.args.ensemble)
        return model_outputs

    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self._step(batch)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def train_epoch(self, i_th: int):
        self.model.train()

        tqdm_train_loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                               desc=f'train epoch{i_th + 1}')

        loss_epoch = []
        for i, batch in tqdm_train_loop:
            loss = self.train_step(batch)
            loss_epoch.append(loss)
            tqdm_train_loop.set_postfix(loss=loss)
        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        return loss_epoch

    def train(self):
        f_best = 0
        for i in range(self.args.max_epochs):
            loss = self.train_epoch(i)
            if self.args.logger == 'wandb':
                wandb.log({'loss': loss})
            print(f"{i + 1} epoch average loss: {loss}")
            if i % 1 == 0:
                p, r, f = self.eval_epoch('dev')
                if f > f_best:
                    f_best = f
                    print(f"f1 achieve best at {i + 1} epoch: {f_best}")
                    path = "best_f1_{:.4f}".format(f_best)
                    self.save(path)

    def eval_step(self, batch):
        bsz = batch[0].shape[0]
        # [bsz, len]
        results, path_x = self._step(batch)
        gold_labels = batch[2]
        pred_labels = results

        labels_mask = gold_labels != -100
        num_gold, num_pred, num_tp = 0, 0, 0
        num_correct_label, num_all_label = 0, 0
        for i in range(bsz):
            gl = gold_labels[i][labels_mask[i]].tolist()
            pl = pred_labels[i][labels_mask[i]].tolist()
            assert len(gl) == len(pl), 'the num of gold and pred labels must be the same'
            gold_ents = self._decode(gl)
            pred_ents = self._decode(pl)
            num_gold += len(gold_ents)
            num_pred += len(pred_ents)
            num_tp += len(list(set(gold_ents).intersection(set(pred_ents))))

            correct_label = sum([1 if g == p else 0 for g, p in zip(gl, pl)])
            num_correct_label += correct_label
            num_all_label += len(gl)

        return num_gold, num_pred, num_tp, num_correct_label, num_all_label

    @torch.no_grad()
    def eval_epoch(self, mode: str):
        dataloader = self.dev_dataloader if mode == 'dev' else self.test_dataloader
        self.model.eval()
        total_gold, total_pred, total_tp = 0, 0, 0
        total_ncl, total_nal = 0, 0
        tqdm_loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'{mode} epoch')
        for i, batch in tqdm_loop:
            ng, np, ntp, ncl, nal = self.eval_step(batch)
            total_gold += ng
            total_pred += np
            total_tp += ntp

            total_ncl += ncl
            total_nal += nal


        print(f"num_gold: {total_gold}")
        print(f"num_pred: {total_pred}")
        print(f"num_true_positive: {total_tp}")
        precision, recall, f1 = self._calculate_prf(total_gold, total_pred, total_tp)
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")
        print(f"label accuracy: {total_ncl / total_nal}")

        prf_table = PrettyTable()
        prf_table.title = "prf per class"
        prf_table.field_names = ['precision', 'recall', 'f1', 'num_pred', 'num_gold', 'num_tp', "label accuracy"]
        prf_table.add_row([precision, recall, f1, total_pred, total_gold, total_tp, total_ncl / total_nal])
        print(prf_table)
        # if self.args.logger == "wandb":
        #     wandb.log({"{} precision": precision})
        #     wandb.log({'recall': recall})
        return precision, recall, f1

    def _calculate_prf(self, num_gold: int, num_pred: int, num_tp: int):
        precision = num_tp / num_pred if num_pred != 0 else 0.
        recall = num_tp / num_gold if num_gold != 0 else 0.
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.

        return precision, recall, f1

    def _decode(self, labels):
        labels = [self.label_set.id2label(i) for i in labels]
        decoded_entities = []

        for i, label in enumerate(labels):
            if label.startswith('S-'):
                decoded_entities.append(((i, i), label[2:]))
            elif label.startswith('B-'):
                start = i
                ent = label[2:]
                j = i + 1
                while j < len(labels):
                    if labels[j] == "M-" + ent:
                        j += 1
                        continue
                    elif labels[j] == "E-" + ent:
                        end = j
                        decoded_entities.append(((start, end), ent))
                        j += 1
                        break
                    else:
                        break

        return decoded_entities

    def save(self, path=None):
        dir_ = '-'.join(self.args.config_file.split('.')[:-1])
        dir_path = os.path.join(self.args.output_dir, dir_)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if path is None:
            path = os.path.join(dir_path, self.args.model_path)
        else:
            path = os.path.join(dir_path, path)
        print(f"save model checkpoints to {path}")
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        dir_ = '-'.join(self.args.config_file.split('.')[:-1])
        dir_path = os.path.join(self.args.output_dir, dir_)
        if path is None:
            path = os.path.join(dir_path, self.args.model_path)
        else:
            path = os.path.join(dir_path, path)
        print(f"load model checkpoints from {path}...")
        self.model.load_state_dict(torch.load(path))
