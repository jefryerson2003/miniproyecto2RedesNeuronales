import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from DataLoader import train_loader, valid_loader  # Importar el DataLoader creado

def main():
    # Definir el dispositivo (CPU o GPU si está disponible)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    # Mover el modelo al dispositivo
    model.to(device)

    # Definir el optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0003)

    # Definir el número de épocas
    num_epochs = 10

    # Función de entrenamiento
    def train_one_epoch(model, optimizer, data_loader, device, epoch):
        model.train()
        total_loss = 0.0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Verificar que las cajas no estén vacías
            if any(t['boxes'].numel() == 0 for t in targets):
                print("Warning: Se encontró un target vacío en el conjunto de datos. Saltando esta imagen.")
                continue

            # Calcular la pérdida
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Retropropagación y optimización
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}], Average Loss: {avg_loss:.4f}")

    # Entrenamiento con manejo de excepciones
    try:
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            train_one_epoch(model, optimizer, train_loader, device, epoch)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), "faster_rcnn_animals.pth")
    print("Entrenamiento completado y modelo guardado como `faster_rcnn_animals.pth`.")

if __name__ == '__main__':
    main()
