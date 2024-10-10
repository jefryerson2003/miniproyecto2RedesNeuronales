# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# Mensaje de inicio
print("Inicializando el cliente de inferencia...")

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="f7jPzAPiQv5SoDs56FpL"
)

print("Cliente inicializado correctamente.")

# Ruta de la imagen para la inferencia
image_path = "AnimalsDataSet-1/test/-1-_jpg.rf.e9e85624c646592d09943c35a774919c.jpg"
model_id = "animalsdataset/1"

# Mensaje de inferencia
print(f"Inferencia en la imagen: {image_path} usando el modelo: {model_id}")

# Realizar inferencia en la imagen local
result = CLIENT.infer(image_path, model_id=model_id)

# Imprimir los resultados de la inferencia
print("Resultados de la inferencia:")
print(result)
