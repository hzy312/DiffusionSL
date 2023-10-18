from trainer_cws_baseline import Trainer as Trainer_cws_baseline
from trainer_cws import Trainer as Trainer_cws
from trainer_ner import Trainer as Trainer_ner
from trainer_ner_baseline import Trainer as Trainer_ner_baseline
from trainer_pos import Trainer as Trainer_pos
from trainer_pos_baseline import Trainer as Trainer_pos_baseline
from options import get_parser
import os


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.base == 'ner':
        trainer = Trainer_ner(args)
    elif args.base == 'cws':
        trainer = Trainer_cws(args)
    elif args.base == 'cws_baseline':
        trainer = Trainer_cws_baseline(args)
    elif args.base == 'ner_baseline':
        trainer = Trainer_ner_baseline(args)
    elif args.base == "pos":
        trainer = Trainer_pos(args)
    elif args.base == 'pos_baseline':
        trainer = Trainer_pos_baseline(args)
    trainer.train()
    trainer.save()
    trainer.eval_epoch('test')


if __name__ == '__main__':
    main()