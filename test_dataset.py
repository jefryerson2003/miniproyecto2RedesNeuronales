import os
import matplotlib.pyplot as plt
from dataset import CustomCocoDataset  # Asegúrate de que el nombre del archivo sea correcto

# Definir las rutas del dataset
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

# Probar que el dataset se carga correctamente
img, target = train_dataset[0]  # Cargar la primera imagen y sus anotaciones

# Mostrar la imagen con las cajas delimitadoras
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img.permute(1, 2, 0))  # Convertir la imagen a formato (Alto, Ancho, Canales)
for box in target['boxes']:
    xmin, ymin, xmax, ymax = box
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
plt.show()
