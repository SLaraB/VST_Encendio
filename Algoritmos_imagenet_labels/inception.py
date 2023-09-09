import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image
import time
import csv
import warnings
warnings.filterwarnings("ignore")


start_time = time.time()

model = torchvision.models.inception_v3(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Obtener el número de características de la capa classifier
num_features = model.fc.in_features


import os
from PIL import Image

folder_path = '../Imagenes/images'  # Ruta a la carpeta que contiene las imágenes

imgs = []  # Lista para almacenar las imágenes

# Recorrer todas las imágenes en la carpeta
for filename in os.listdir(folder_path):
    # Comprobar si el archivo es una imagen
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpeg'):
        # Abrir la imagen y agregarla a la lista imgs
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        imgs.append(img)



# In[8]:


# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''
def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))


# In[4]:


# Aplicar las funciones a todas las imágenes en la lista imgs
processed_imgs = []
for img in imgs:
    processed_img = preprocess(img)
    processed_imgs.append(processed_img)



# In[30]:


import torch
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Función para redimensionar la imagen de saliencia al mismo tamaño que la imagen original
def resize_saliency_map(saliency_map, target_size):
    saliency_map = torch.squeeze(saliency_map)  # Eliminar la dimensión adicional
    saliency_map = F.interpolate(saliency_map.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    saliency_map = torch.squeeze(saliency_map)
    return saliency_map

# Se asume que 'model' ya está definido y se ha cargado el modelo pre-entrenado.

# Se inicializa la lista para almacenar las imágenes originales y los mapas de saliencia
original_images = []
saliency_maps = []

# Se establece el modelo en modo de evaluación
model.eval()

for img in imgs:
    # Preprocesamiento de la imagen
    X = preprocess(img)
    
    # Se requiere el cálculo del gradiente con respecto a la imagen de entrada
    X.requires_grad_()
    
    # Forward pass a través del modelo para obtener las puntuaciones
    scores = model(X)
    
    # Obtener el índice y el valor máximo de la puntuación
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    
    # Retropropagación para calcular los gradientes
    score_max.backward()
    
    # Obtener el mapa de saliencia
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    
    # Redimensionar la imagen de saliencia al tamaño de la imagen original
    saliency_resized = resize_saliency_map(saliency, img.size[::-1])
    
    # Añadir la imagen original y el mapa de saliencia redimensionado a las listas correspondientes
    original_images.append(img)
    saliency_maps.append(saliency_resized)

# Visualizar las imágenes originales y los mapas de saliencia
'''
for i in range(len(imgs)):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_images[i])
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    axes[1].imshow(saliency_maps[i], cmap=plt.cm.hot)
    axes[1].axis('off')
    axes[1].set_title('Saliency Map')
    plt.show()
'''



# In[42]:


import numpy as np
import os

# Obtener la lista de etiquetas de clase
class_labels = model.fc.in_features
labels_path = "../imagenet_labels.txt"  # Ruta al archivo de etiquetas de clase
with open(labels_path) as f:
    class_labels = f.read().splitlines()

# Obtener la ruta completa de la carpeta de resultados
resultados_folder = 'Resultados_Inception'  # Ruta a la carpeta de resultados

# Verificar si la carpeta de resultados existe, de lo contrario, crearla
if not os.path.exists(resultados_folder):
    os.makedirs(resultados_folder)

# Obtener la ruta completa del archivo CSV en la carpeta "CSVs"
csv_folder = 'CSVs'
csv_filename = os.path.join(csv_folder, 'inception_results.csv')

# Verificar si la carpeta "CSVs" existe, de lo contrario, crearla
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Verificar si el archivo CSV ya existe
file_exists = os.path.exists(csv_filename)

# Abrir el archivo CSV en modo de escritura y crearlo si no existe
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['Saliency Map Path', 'Label'])

    for i in range(len(imgs)):
        # Obtener la imagen original y el mapa de saliencia correspondiente
        original_img = original_images[i]
        saliency_map = saliency_maps[i]

        # Ajustar el tamaño del mapa de saliencia al tamaño de la imagen original
        saliency_map_resized = resize_saliency_map(saliency_map, original_img.size[::-1])

        # Normalizar el mapa de saliencia
        saliency_map_normalized = (saliency_map_resized - saliency_map_resized.min()) / (saliency_map_resized.max() - saliency_map_resized.min())

        # Definir el umbral para determinar las regiones destacadas
        threshold = 0.05

        # Obtener el índice y el valor máximo de la puntuación
        scores = model(preprocess(original_img))
        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]

        # Obtener la etiqueta de clase correspondiente al índice
        class_label = class_labels[score_max_index]

        # Convertir el mapa de saliencia a un arreglo NumPy
        saliency_map_np = saliency_map_normalized.cpu().numpy()

        # Crear una máscara utilizando el mapa de saliencia
        mask = np.ones_like(original_img)
        mask[saliency_map_np < threshold] = 0  # Establecer a cero las regiones no destacadas

        # Aplicar la máscara a la imagen original
        highlighted_img = original_img * mask

        # Mostrar la imagen original, el mapa de saliencia y la imagen reconstituida
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_img)
        axes[0].axis('off')
        axes[0].set_title(f'Original Image (Object: {class_label})')

        axes[1].imshow(saliency_map_normalized, cmap=plt.cm.hot)
        axes[1].axis('off')
        axes[1].set_title('Saliency Map')

        axes[2].imshow(highlighted_img)
        axes[2].axis('off')
        axes[2].set_title('Highlighted Image')

        plt.tight_layout()
        # Guardar la figura en la carpeta de resultados
        plt.savefig(os.path.join(resultados_folder, f'image_figures_{i}.png'))
        #print(f"Guardada imagen en: {os.path.join(resultados_folder, f'image_figures_{i}.png')}")
        # Cerrar la figura para liberar recursos
        plt.close(fig)

        saliency_map_filename = f'image_figures_{i}.png'

        # Escribir la información en una nueva fila del CSV
        writer.writerow([saliency_map_filename,class_label ])

end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución Inception: {execution_time} segundos")