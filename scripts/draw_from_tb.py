import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_item_with_label(ea: event_accumulator.EventAccumulator, label: str):
    data = ea.scalars.Items(label)
    plt.plot([i.step for i in data],
             [i.value for i in data], label=label)


def main(args):

    ea = event_accumulator.EventAccumulator(
        glob(os.path.join(args.experiment_path, '*.tfevents.*'))[0])

    ea.Reload()

    plot_item_with_label(ea, 'train/loss')
    plot_item_with_label(ea, 'val/loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(args.experiment_path, 'loss.png'))

    plot_item_with_label(ea, 'train/acc1')
    plot_item_with_label(ea, 'val/acc1')
    plt.legend()
    plt.title('Acc1')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(args.experiment_path, 'acc1.png'))

    plot_item_with_label(ea, 'train/acc5')
    plot_item_with_label(ea, 'val/acc5')
    plt.legend()
    plt.title('Acc5')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(args.experiment_path, 'acc1.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-path')

    args = parser.parse_args()

    main(args)
