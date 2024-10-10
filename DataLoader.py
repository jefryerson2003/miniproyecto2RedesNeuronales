import os
import torch
from torch.utils.data import DataLoader, Subset
from dataset import CustomCocoDataset  # Asegúrate de que esta sea tu clase de dataset personalizada

# Definir la función collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# Definir las rutas de los datasets
train_dir = "AnimalsDataSet-1/train"
valid_dir = "AnimalsDataSet-1/valid"

# Crear las instancias del dataset personalizado
train_dataset = CustomCocoDataset(
    img_folder=train_dir, 
    annotation_file=os.path.join(train_dir, "_annotations.coco.json")
)
valid_dataset = CustomCocoDataset(
    img_folder=valid_dir, 
    annotation_file=os.path.join(valid_dir, "_annotations.coco.json")
)

# Crear subconjuntos para el DataLoader
subsample_size = 5  # Ajusta este número según lo que necesites
train_indices = list(range(subsample_size))
valid_indices = list(range(subsample_size))  # Usa el mismo tamaño o ajústalo como desees

# Crear subconjuntos
train_subset = Subset(train_dataset, train_indices)
valid_subset = Subset(valid_dataset, valid_indices)

# Crear los DataLoaders para entrenamiento y validación
num_workers = 2  # Ajustar según el número de núcleos de tu CPU

train_loader = DataLoader(
    train_subset,  # Usa el subconjunto de entrenamiento
    batch_size=2, 
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=num_workers,
    pin_memory=True  # Habilitar el pinning de memoria
)

valid_loader = DataLoader(
    valid_subset,  # Usa el subconjunto de validación
    batch_size=2, 
    shuffle=False, 
    collate_fn=collate_fn, 
    num_workers=num_workers,
    pin_memory=True  # Habilitar el pinning de memoria
)
