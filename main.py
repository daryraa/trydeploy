from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json, pickle, random
import onnxruntime as ort
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# === Inisialisasi FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CHATBOT SECTION ====================
interpreter = tf.lite.Interpreter(model_path="chatbot_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(json.load(f))

with open("label_encoder.json", "r") as f:
    label_encoder = json.load(f)
le = LabelEncoder()
le.classes_ = np.array(label_encoder)

with open("chatbot-intents-variasi-unik.json", "r", encoding="utf-8") as f:
    data = json.load(f)

MAX_LEN = input_details[0]['shape'][1]

class InputData(BaseModel):
    text: str

def predict_tflite(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    padded = np.array(padded, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output, axis=1)[0])
    confidence = float(np.max(output))
    predicted_label = le.inverse_transform([predicted_index])[0]
    return predicted_label, confidence

@app.post("/predict")
async def predict(input: InputData):
    label, prob = predict_tflite(input.text)
    for intent in data["intents"]:
        if intent["tag"] == label:
            return {
                "tag": label,
                "confidence": float(prob),
                "response": random.choice(intent["responses"])
            }
    return {
        "tag": label,
        "confidence": float(prob),
        "response": "Maaf kak, aku belum bisa menjawab itu 😢"
    }

# ==================== STUNTING SECTION ====================
stunting_model = ort.InferenceSession("gbc_model.onnx")
stunting_input = stunting_model.get_inputs()[0].name

category_mapping = {
    'Jenis kelamin': {'Laki-laki': 0, 'Perempuan': 1},
    'Apakah ibu dan anak melakukan pemeriksaan setelan 40 hari kelahiran?': {'Tidak': 0, 'Ya': 1},
    'Tingkat pendidikan kepala keluarga': {
        'Tingkat SD': 2, 'Tingkat SMP': 4, 'Pendidikan Tinggi': 0, 'Tingkat SMA': 3, 'Tidak sekolah': 1
    }
}

target_map = {0: "Normal", 1: "Stunting"}

def preprocess_input_stunting(raw_inputs):
    feature_order = [
        'Jenis kelamin', 'Usia anak (bulan)', 'Berat badan anak saat ini (kg)', 'Tinggi badan anak saat ini (cm)',
        'Apakah ibu dan anak melakukan pemeriksaan setelan 40 hari kelahiran?', 'Berat badan anak saat lahir (kg)',
        'Usia anak pertama kali disapih (bulan)', 'Jumlah pemeriksaan yang dilakukan ibu saat hamil',
        'Frekuensi anak makan ubi dalam 1 minggu', 'Frekuensi anak makan telur dalam 1 minggu',
        'Frekuensi anak makan ikan dalam 1 minggu', 'Frekuensi anak makan daging dalam 1 minggu',
        'Frekuensi anak minum susu dalam 1 minggu', 'Frekuensi anak makan sayur dalam 1 minggu',
        'Frekuensi anak makan pisang dalam 1 minggu', 'Frekuensi anak makan pepaya dalam 1 minggu',
        'Frekuensi anak makan wortel dalam 1 minggu', 'Frekuensi anak makan mangga dalam 1 minggu',
        'Frekuensi anak makan mie dalam 1 minggu', 'Frekuensi anak makan fast food dalam 1 minggu',
        'Frekuensi anak minum soda dalam 1 minggu', 'Frekuensi anak makan sambal dalam 1 minggu',
        'Frekuensi anak makan gorengan dalam 1 minggu', 'Frekuensi anak makan nasi dalam 1 minggu',
        'Frekuensi anak makan makanan manis dalam 1 minggu', 'Tingkat pendidikan kepala keluarga',
        'Pendapatan keluarga perbulan (Rp)'
    ]
    inputs = []
    for f in feature_order:
        v = raw_inputs.get(f)
        if f in category_mapping:
            mapped_val = category_mapping[f].get(v)
            if mapped_val is None:
                raise ValueError(f"Nilai '{v}' tidak valid untuk fitur kategori '{f}'")
            inputs.append(float(mapped_val))
        else:
            inputs.append(float(v))
    return inputs

def predict_stunting(raw_inputs):
    input_array = np.array([preprocess_input_stunting(raw_inputs)], dtype=np.float32)
    preds = stunting_model.run(None, {stunting_input: input_array})
    predicted_class = preds[0][0]
    predicted_prob = preds[1][0][1]
    return target_map.get(predicted_class, "Unknown"), predicted_prob

class StuntingInput(BaseModel):
    inputs: dict

@app.post("/predict_stunting")
async def predict_stunting_api(input: StuntingInput):
    label, prob = predict_stunting(input.inputs)
    return {
        "prediction": label,
        "probability": float(prob)
    }

# ==================== KEK SECTION ====================
with open("preprocessing_params.json", "r") as f:
    kek_preproc = json.load(f)

kek_model = ort.InferenceSession("xgb_model.onnx")
kek_input_name = kek_model.get_inputs()[0].name
kek_expected_features = kek_model.get_inputs()[0].shape[1]

numerical_columns = [
    'Umur Ibu (tahun)', 'TB (cm)', 'Jarak Hamil', 'Tinggi Fundus Uteri (TFU)', 'Detak Jantung Janin',
    'Pemeriksaan HB', 'Panjang BBL (cm)', 'Berat BBL (gr)', 'Sistol', 'Diastol',
    'Gravida', 'Para', 'Abortus', 'Usia Kehamilan Minggu'
]

categorical_columns = [
    'Jenis Asuransi', 'IMT Sebelum Hamil', 'Status Td', 'Presentasi', 'Gol Darah dan Rhesus',
    'Rujuk Ibu Hamil', 'Faskes Rujukan', 'Konseling', 'Komplikasi', 'Cara Persalinan',
    'Tempat Bersalin', 'Penolong Persalinan', 'Kondisi Ibu', 'Kondisi Bayi',
    'Komplikasi Persalinan', 'Rujuk Ibu Bersalin (Ya / Tidak)', 'Komplikasi Masa Nifas',
    'Rujuk Ibu Nifas', 'Kelurahan/Desa'
]

def preprocess_input_kek(user_input):
    means = kek_preproc["numerical_imputer"]["mean"]
    encoder_cats = kek_preproc["categorical_encoder"]["categories"]

    X_num = [float(user_input.get(col, means[i])) if user_input.get(col) not in ["", "NaN", None] else means[i]
             for i, col in enumerate(numerical_columns)]
    X_cat = []
    for i, col in enumerate(categorical_columns):
        val = user_input.get(col, "")
        valid_cats = encoder_cats[i]
        onehot = [1.0 if val == cat else 0.0 for cat in valid_cats]
        if sum(onehot) == 0:
            onehot = [0.0] * len(valid_cats)
        X_cat.extend(onehot)

    X = np.concatenate([np.array(X_num).reshape(1, -1), np.array(X_cat).reshape(1, -1)], axis=1)
    if X.shape[1] < kek_expected_features:
        X = np.concatenate([X, np.zeros((1, kek_expected_features - X.shape[1]))], axis=1)
    elif X.shape[1] > kek_expected_features:
        X = X[:, :kek_expected_features]
    return X.astype(np.float32)

class KEKInput(BaseModel):
    inputs: dict

@app.post("/predict_kek")
async def predict_kek_api(input: KEKInput):
    X = preprocess_input_kek(input.inputs)
    pred = kek_model.run(None, {kek_input_name: X})
    pred_label = int(np.argmax(pred[0]))
    confidence = float(np.max(pred[0][0]))  # ✅ Ambil confidence tertinggi

    label_dict = {0: "KEK", 1: "Normal", 2: "Resiko KEK"}
    return {
        "prediction": label_dict.get(pred_label, "Unknown"),
        "probability": confidence 
    }
