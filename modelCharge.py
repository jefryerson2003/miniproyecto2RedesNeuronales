import torch

# Cargar el modelo
model_weights = torch.load("faster_rcnn_animals.pth")

# Función para imprimir los pesos de manera ordenada
def print_weights(weights):
    for name, value in weights.items():
        # Mostrar solo los primeros 5 elementos para evitar salidas demasiado largas
        print(f"Nombre: {name}")
        print(f"Forma: {value.shape}")
        print("Valores (primeros 5 elementos):")
        print(value.flatten()[:5])  # Muestra los primeros 5 elementos del tensor
        print("-" * 40)

# Llamar a la función con los pesos del modelo
print_weights(model_weights)
