import streamlit as st
import numpy as np
import keras
import librosa
import sounddevice as sd
import io
from scipy.io.wavfile import write

model = keras.models.load_model('../models/gru_model_split552.h5')

st.title('Audio Language Classification')



def record_audio(duration=5, sr=16000):
    recording = sd.rec(frames=duration * sr, samplerate=sr, channels=1).reshape(-1)
    sd.wait()
    return recording

def predict_language(mfcc):
    pred_labels = ['English', 'Spanish', 'French', 'Russian', 'Mandarin Chinese']
    pred = model.predict(np.array([mfcc])).T
    labeled = list(zip(pred_labels, pred))
    labeled.sort(key=lambda x: x[1], reverse=True)

    for lang, dec in labeled:
        pct = np.round(dec[0] * 100, 2)
        st.write(f'{lang} : {pct}%')

if st.button('Start Recording'):
    with st.spinner('Recording...'):
        recording = record_audio()
    
    temp_file = io.BytesIO()
    write(temp_file, 16000, recording)
    st.audio(temp_file, format='audio/wav')

    mfcc = librosa.effects.feature.mfcc(recording, n_mfcc=10)

    st.write('Audio Frequency')
    st.line_chart(recording)
    
    st.write('Mel Frequency Cepstrum Coefficients')
    st.line_chart(mfcc.T)

    st.write('Model Prediction')
    predict_language(mfcc)


