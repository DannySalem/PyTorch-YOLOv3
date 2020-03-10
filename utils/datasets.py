import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        #self.to_pad = True
        #self.to_crop = False
        self.to_pad = False
        self.to_crop = True
        self.crop_size = 500

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # Pad to square resolution
        if self.to_pad:
            _, h, w = img.shape
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            img, pad = pad_to_square(img, 0)
            _, padded_h, padded_w = img.shape


        # Get Crop Bounds
        if self.to_crop:
            _, h, w = img.shape
            topleft_x = random.randint(0, w-self.crop_size-1)
            topleft_y = random.randint(0, h-self.crop_size-1)
            #w_offset = (w - self.crop_size) / 2
            top_bound = topleft_y
            bottom_bound = top_bound + self.crop_size
            left_bound = topleft_x
            right_bound = left_bound + self.crop_size
            cropped_img = img[:,top_bound:bottom_bound, left_bound:right_bound]
        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):

            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            targets = []

            if self.to_crop:
                for box in boxes:
                    x1 = int((box[1] - box[3] / 2) * w)
                    y1 = int((box[2] - box[4] / 2) * h)
                    x2 = int((box[1] + box[3] / 2) * w)
                    y2 = int((box[2] + box[4] / 2) * h)
                    if x1 > left_bound and x2 < right_bound and y1 > top_bound and y2 < bottom_bound:
                        # Re-normalize to cropped image
                        x1 = abs(x1 - (left_bound)) / self.crop_size
                        x2 = abs(x2 - (left_bound)) / self.crop_size
                        y1 = abs(y1 - (top_bound)) / self.crop_size
                        y2 = abs(y2 - (top_bound)) / self.crop_size
                        # Make output
                        box_out = torch.zeros(6)
                        box_out[1] = box[0]
                        box_out[2] = (x1 + x2) / 2
                        box_out[3] = (y1 + y2) / 2
                        box_out[4] = x2 - x1
                        box_out[5] = y2 - y1
                        targets += [box_out]
                targets = torch.stack(targets) if len(targets) > 0 else None
                img = cropped_img

            if self.to_pad:
                targets = torch.zeros((len(boxes), 6))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                # Returns (x, y, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h
                targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    # Custom Batching function. Dataset can't output a batch if images are different resolution
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class TestDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.batch_count = 0

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):

            boxes = torch.from_numpy(np.loadtxt(label_path, usecols=(0, 1, 2, 3, 5), delimiter=',').reshape(-1, 5))
            w = boxes[:, 2]
            h = boxes[:, 3]
            boxes[:, 2] = boxes[:, 0] + w
            boxes[:, 3] = boxes[:, 1] + h
            targets = torch.zeros((len(boxes), 7))
            targets[:, 0:4] = boxes[:, 0:4]
            targets[:, 6] = boxes[:, 4]
            targets[:, 4] = torch.ones(1, (len(boxes)))
            targets[:, 5] = torch.ones(1, (len(boxes)))
        return img_path, targets

    def __len__(self):
        return len(self.img_files)
