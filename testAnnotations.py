from __future__ import division
from utils.utils import load_classes, printBBoxes
from utils.datasets import TestDataset
from utils.parse_config import parse_data_config
import torch


if __name__ == "__main__":

    epochs = 100
    model_def = 'config/yolov3visdrone.cfg'
    data_config = 'config/visdrone.data'
    data_config = parse_data_config(data_config)
    class_names = load_classes(data_config["names"])
    train_path = data_config["train"]
    valid_path = data_config["valid"]

    # Get dataloader
    dataset = TestDataset(
        train_path
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,  # opt.n_cpu,
        pin_memory=True
    )

    for epoch in range(epochs):

        for batch_i, (paths, targets) in enumerate(dataloader):
            printBBoxes(paths[0], targets[0], class_names, rescale=False)
