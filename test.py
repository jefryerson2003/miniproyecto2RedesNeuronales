import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Definir el número de clases (incluye 1 clase de fondo)
num_classes = 6  # 5 clases de animales + 1 para el fondo

# Usar ResNet50 como el backbone preentrenado para Faster R-CNN
backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1")
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 2048

# Crear la AnchorGenerator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# Definir el ROI pooler
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=["0"],
    output_size=7,
    sampling_ratio=2
)

# Crear el modelo Faster R-CNN
model = FasterRCNN(
    backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

# Cargar los pesos del modelo
model.load_state_dict(torch.load("faster_rcnn_animals.pth"))  # No usar weights_only aquí
model.eval()  # Poner el modelo en modo de evaluación

# Mapeo de etiquetas
label_map = {
    1: "Bird",
    2: "Cat",
    3: "Dog",
    4: "Hamster",
    5: "Horse",
    0: "Background"  # Asegúrate de incluir el fondo
}

# Función para cargar y procesar la imagen
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # Convertir la imagen a un tensor
        T.Resize((800, 800)),  # Redimensionar si es necesario
    ])
    return transform(image).unsqueeze(0)  # Añadir una dimensión para el batch

# Función para hacer predicciones
def predict(image_tensor, model):
    with torch.no_grad():  # No necesitamos calcular gradientes
        predictions = model(image_tensor)
    return predictions

# Función para visualizar y guardar los resultados
def visualize_predictions(image_path, predictions, threshold=0.2):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Obtener cajas, etiquetas y puntajes
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # Filtrar predicciones por umbral
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:  # Solo mostrar si la puntuación es mayor que el umbral
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
            class_name = label_map.get(label.item(), "Unknown")  # Obtener el nombre de la clase
            plt.text(x1, y1, f'{class_name}: {score.item():.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.savefig('Hamster.png')  
    plt.close()  
    print("Figura guardada como 'hamster.png'")
   

# Ruta de la imagen de prueba
image_path = 'AnimalsDataSet-1/test/-31-_jpg.rf.ac984d0f516d05871dc26f835d8b8824.jpg'  # Cambia a la ruta correcta de tu imagen

# Cargar la imagen
image_tensor = load_image(image_path)

# Hacer la predicción
predictions = predict(image_tensor, model)

# Visualizar las predicciones
visualize_predictions(image_path, predictions)
