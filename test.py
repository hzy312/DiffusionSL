
from trainer_cws import Trainer as Trainer_cws
from trainer_ner import Trainer as Trainer_ner
from trainer_pos import Trainer as Trainer_pos

from options import get_parser
import os
from utils import ensure_reproducibility
import matplotlib.pyplot as plt

# def record(trainer: Trainer, steps: int = 10):
#     f_per_step = []
#     for i in range(1, 1 + steps):
#         trainer.model.timesteps = i
#         _, _, f = trainer.eval_epoch('test')
#         f_per_step.append(f)
#     print(f_per_step)
#     plt.plot(f_per_step)
#     plt.show()



def main():
    ensure_reproducibility(3407)
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
    elif args.base == 'pos':
        trainer = Trainer_pos(args)
    trainer.load("best_f1_0.9658")
    trainer.eval_epoch('test')
    
    # trainer.eval_path('test')
    # record(trainer, 30)

if __name__ == '__main__':
    main()
