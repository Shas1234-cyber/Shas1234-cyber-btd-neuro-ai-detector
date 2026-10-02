import tensorflow as tf
from tensorflow.keras.models import load_model


model_path = 'brain_tumor_vgg16_90acc.keras'
model = load_model(model_path)

print("Model successfully loaded!")

model.summary()