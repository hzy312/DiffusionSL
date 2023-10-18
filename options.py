"""
@Date  : 2022/12/18
@Time  : 15:18
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import argparse
import yaml
import os


def get_parser():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--base", default='ner')
    parser.add_argument("--config_file", default="resume.bert-base-uncased.ner.yaml", type=str)
    parser.add_argument("--dataset", default="fewnerd-few_nerd", type=str)
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--backbone", default='../plm/bert-base-chinese', type=str)
    parser.add_argument("--time_steps", default=1000, type=int)
    parser.add_argument("--sampling_steps", default=10, type=int)
    parser.add_argument("--ddim_sampling_eta", default=1., type=float)
    parser.add_argument("--self_condition", default=False, type=bool)
    parser.add_argument("--snr_scale", default=2., type=float)
    parser.add_argument("--dim_model", default=768, type=int)
    parser.add_argument("--encoder_depth", default=3, type=int, dest="the depth of tranformer encoder")
    parser.add_argument("--decoder_depth", default=6, type=int, dest="the depth of tranformer decoder")
    parser.add_argument("--dim_time", default=256, type=int)
    parser.add_argument("--objective", default='pred_x0', type=str)
    parser.add_argument("--noise_schedule", default="linear", type=str)
    parser.add_argument("--loss_type", default='l2', choices=['l1', 'l2'])
    parser.add_argument("--add_lstm", default=False, type=bool)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--decode_mode", default='bmes', type=str)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--network_architecture", default="transformer", type=str)
    parser.add_argument("--ensemble", default=False, type=bool)

    # training args
    parser.add_argument("--logger", default='None', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--model_path", default='model.pt', type=str)
    parser.add_argument("--use_gpu", default=False, type=bool)
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--max_steps", default=250000, type=int)
    parser.add_argument("--max_epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=6, type=int, dest="num_workers for dataloader, 0 for debugging")
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--warmup_ratio", default=0.01, type=float)
    parser.add_argument("--optimizer_type", default='AdamW', type=str)
    parser.add_argument("--lr_scheduler_type", default='linear', type=str)
    parser.add_argument("--num_cycles", default=1, type=int)
    parser.add_argument("--lr_bert", default=5e-5, type=float)
    parser.add_argument("--lr_other", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--accumulation_steps", default=4, type=int)
    parser.add_argument("--test_path", action='store_true')
    args = parser.parse_args()
    default_path = os.path.join(os.getcwd(), "configs", args.config_file)
    with open(default_path, 'r') as f:
        default_args_from_file = yaml.load(f, Loader=yaml.FullLoader)
    parser.set_defaults(**default_args_from_file)

    return parser