import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.datasets import CocoDetection

# Crear una clase personalizada para cargar las imágenes y etiquetas del dataset en formato COCO
class CustomCocoDataset(CocoDetection):
    def __init__(self, img_folder, annotation_file, transforms=None):
        super(CustomCocoDataset, self).__init__(img_folder, annotation_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        # Cargar la imagen y sus anotaciones usando la clase base CocoDetection
        img, target = super(CustomCocoDataset, self).__getitem__(idx)

        # Convertir la imagen a tensor
        img = torchvision.transforms.ToTensor()(img)

        # Adaptar el formato de las anotaciones al formato que Faster-RCNN espera
        boxes = []
        labels = []

        for obj in target:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            width = obj['bbox'][2]
            height = obj['bbox'][3]
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        # Convertir a tensores de PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Crear un diccionario con los elementos esperados por el modelo
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
