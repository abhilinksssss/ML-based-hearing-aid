# ML-based-hearing-aid
ML based aid for the deaf that detect vibrations for sounds data collection

import os
import librosa
import numpy as np
import pandas as pd

# Sound categories
classes = ['baby', 'horn', 'dog', 'doorbell', 'alarm']
data = []

def extract_features(file_path, label):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean, label

# Simulate data
root_dir = 'sounds/'  # organized as sounds/baby/xxx.wav, sounds/horn/xxx.wav, etc.
for label in classes:
    folder = os.path.join(root_dir, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            feat, lbl = extract_features(os.path.join(folder, file), label)
            data.append(np.append(feat, lbl))

# Save features
df = pd.DataFrame(data)
df.to_csv("vibration_features.csv", index=False)

#train model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load features
df = pd.read_csv("vibration_features.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "hearing_aid_model.pkl")

# simulation
import joblib
import librosa
import numpy as np

# Load model
model = joblib.load("hearing_aid_model.pkl")

# Simulate live audio (or vibration recording)
def predict_sound(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    pred = model.predict(mfcc_mean)
    return pred[0]

# Test
sound_file = "realtime_input.wav"
print("Detected Sound:", predict_sound(sound_file))