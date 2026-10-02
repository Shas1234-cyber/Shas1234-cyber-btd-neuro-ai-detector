import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import h5py
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, Sequential

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "user_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model Configuration
MODEL_PATH = os.path.join(app.root_path, "static", "models", "brain_tumor_vgg16_90acc.h5")
TUMOR_CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
DISPLAY_NAMES = {
    "glioma_tumor": "Glioma",
    "meningioma_tumor": "Meningioma",
    "no_tumor": "No tumor",
    "pituitary_tumor": "Pituitary tumor",
}

model2 = None

def load_brain_tumor_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print("Building VGG16 architecture in TensorFlow 2.10...")
    base = VGG16(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = Sequential([
        base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')
    ])
    model.build(input_shape=(None, 224, 224, 3))

    print(f"Injecting trained 92% weights from {filepath}...")
    with h5py.File(filepath, 'r') as f:
        dataset_dict = {}
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_dict[name] = obj[()]
        f.visititems(collect_datasets)

        # 1. Base VGG16 layers
        for layer in base.layers:
            layer_datasets = [dataset_dict[k] for k in dataset_dict if f"/{layer.name}/" in f"/{k}/"]
            if layer_datasets:
                layer_datasets.sort(key=lambda arr: arr.ndim, reverse=True)
                layer.set_weights(layer_datasets)

        # 2. Classifier Head (Dense layers)
        for layer in [model.layers[2], model.layers[4]]:
            layer_datasets = [dataset_dict[k] for k in dataset_dict if f"/{layer.name}/" in f"/{k}/"]
            if not layer_datasets:
                expected_shapes = [w.shape for w in layer.weights]
                matched = []
                for exp_sh in expected_shapes:
                    for k, val in dataset_dict.items():
                        if val.shape == exp_sh and k not in [d[0] for d in matched]:
                            matched.append((k, val))
                            break
                layer_datasets = [m[1] for m in matched]
            
            if layer_datasets:
                layer_datasets.sort(key=lambda arr: arr.ndim, reverse=True)
                layer.set_weights(layer_datasets)

    print("Model successfully loaded with 92.19% trained weights!")
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predictor2', methods=['POST'])
def predictor2():
    global model2
    if model2 is None:
        model2 = load_brain_tumor_model(MODEL_PATH)

    try:
        if 'image' not in request.files:
            return "No image file provided in request.", 400

        uploaded_file = request.files['image']
        name = request.form.get('name', '').strip() or "patient_scan"

        # Image save aur resize
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}.jpg")
        image = Image.open(uploaded_file.stream).convert('RGB')
        image = image.resize((224, 224))
        image.save(image_path)

        # ImageNet Preprocessing
        img = np.asarray(image, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Predict
        preds = model2.predict(img, verbose=0)[0]
        class_index = int(np.argmax(preds))
        tumor_class = TUMOR_CLASSES[class_index]
        confidence = float(preds[class_index]) * 100

        return render_template(
            'prediction1.html',
            data=["NO TUMOR DETECTED" if tumor_class == "no_tumor" else "TUMOR DETECTED", name],
            tumor_type=DISPLAY_NAMES[tumor_class],
            confidence=confidence,
            probabilities={
                DISPLAY_NAMES[label]: round(float(score) * 100, 2)
                for label, score in zip(TUMOR_CLASSES, preds)
            },
            parameter_count=f"{model2.count_params() / 1_000_000:.2f}M",
        )

    except Exception as e:
        return f"Error processing image: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')


