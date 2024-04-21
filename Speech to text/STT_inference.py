"""
Prerequisites:
tensorflow

How to use this file 
1. Call main(model_filepath, wav_filepath) function  

"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

frame_length = 256
frame_step = 160

ch = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
stoi = keras.layers.StringLookup(vocabulary=ch, oov_token="")
itos = keras.layers.StringLookup(vocabulary=stoi.get_vocabulary() , oov_token="", invert=True)

# Function required by models
@keras.saving.register_keras_serializable()
def CTC_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Encode and decode functions
def encodeSample(wavFile):
    wav = tf.io.read_file(wavFile)
    wav, _ = tf.audio.decode_wav(wav)
    wav = tf.squeeze(wav, axis=-1)
    audio = tf.cast(wav, tf.float32)
    spectogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spectogram = tf.abs(spectogram)
    spectogram = tf.math.pow(spectogram, 0.5)
    means = tf.math.reduce_mean(spectogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectogram, 1, keepdims=True)
    spectogram = (spectogram - means) / stddevs + 1e-9
    spectogram = spectogram.numpy()
    spectogram = tf.expand_dims(spectogram, 0)
    return spectogram

def decodeSample(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length = input_len, greedy = True)[0][0]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(itos(res)).numpy().decode("utf-8")
        output_text.append(res)

    return output_text

# Main function
def main(model, wavFile):
    model = keras.models.load_model(model)
    spectogram = encodeSample(wavFile)
    pred = model.predict(spectogram)
    return decodeSample(pred)