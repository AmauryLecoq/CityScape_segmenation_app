# Vérifier l'installation : pip install fastapi uvicorn

# 1. Import des packages
import uvicorn

import numpy as np

from fastapi import FastAPI, UploadFile, File

from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.responses import Response


from typing import Callable, Union

import uuid

import tensorflow as tf
from keras import backend as K

from pathlib import Path


import cv2

# 2. Création de l'objet app et import model
app = FastAPI()

def multiclass_weighted_tanimoto_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted Tanimoto loss.
    Defined in the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data",
    under 3.2.4. Generalization to multiclass imbalanced problems. See https://arxiv.org/pdf/1904.00592.pdf
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Tanimoto loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute weighted Tanimoto loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Tanimoto loss (tf.Tensor, shape=(None, ))
        """
        axis_to_reduce = range(1, K.ndim(y_pred))  # All axis but first (batch)
        numerator = y_true * y_pred * class_weights
        numerator = K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2 - y_true * y_pred) * class_weights
        denominator = K.sum(denominator, axis=axis_to_reduce)
        return 1 - numerator / denominator

    return loss


def multiclass_weighted_dice_score(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)

        return numerator / denominator

    return score

train_class_weight = [1.4043728512699085,
 0.6968967529124955,
 1.289214697933187,
 26.06381506763811,
 9.470404941148304,
 24.019021785775,
 116.41321720021769,
 12.615900788485364]

color_map = { '0': [0, 0, 0],
 '1': [128, 64,128],
 '2': [70, 70, 70],
 '3': [153,153,153],
 '4': [107,142, 35],
 '5': [70,130,180],
 '6': [220, 20, 60],
 '7': [0,  0,142]
}
alpha = 0.6

BASE_DIR = Path(__file__).resolve(strict=True).parent
#model_dir = "export_model"
#with tf.device('/cpu:0'):
"""export_model = tf.keras.models.load_model(model_dir,
                                          custom_objects={'loss':multiclass_weighted_tanimoto_loss(train_class_weight),
                                                          'score':multiclass_weighted_dice_score(train_class_weight)
                                                         })"""
model_dir= f"{BASE_DIR}/export_model"
export_model = tf.keras.models.load_model(model_dir,
                                          custom_objects={'loss':multiclass_weighted_tanimoto_loss(train_class_weight),
                                                          'score':multiclass_weighted_dice_score(train_class_weight)
                                                         })

# 3. Index route, s'ouvre automatiquement sur http://127.0.0.1:8000
@app.get('/')
def index():
    return {"Bienvenue sur l'app de prédiction"}

# 4. Préparation de la page de prédiction

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    dims = img_original.shape
    
    
    img_color = img_original.copy()
    img = cv2.resize(img_original, (224, 224))
    
    img = np.float32(img)/255
    img_dimensions = str(img.shape)
    #with tf.device('/cpu:0'):
    result = export_model.predict(np.expand_dims(img, axis=0))
    
    result = np.squeeze(result)
    result = result.reshape(224, 224, 8)
    
    result = cv2.resize(result, (dims[1], dims[0]))
    
    result = np.argmax(result, axis=2)
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            img_color[i, j] = color_map[str(result[i, j])]
    
    #Pour Debug
    #img_dimensions = str(img.shape)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    success, encoded_img = cv2.imencode('.png', img_color)
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(encoded_img.tobytes() , headers=headers, media_type='image/png')

    #Pour debug
    #img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #success, encoded_img = cv2.imencode('.png', img_color)
    #headers = {'Content-Disposition': 'inline; filename="test.png"'}
    #return Response(encoded_img.tobytes() , headers=headers, media_type='image/png')

# Lancer l'api avec unicorn
# sera lancer sur http://127.0.0.1:8000

if __name__ == '__main__':
    uvicorn.run(app, host ="0.0.0.0", port=8000)
    #uvicorn.run(app, host ="127.0.0.1:8000", port=8000)

# uvicorn <nom du fichier actuel>:app --reload