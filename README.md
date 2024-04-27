# Video Dubbing with ML-Driven Lip Synchronization

This project aims to develop a machine learning-based solution for dubbing videos from one language to another, addressing challenges such as precise lip synchronization, and natural pauses between sentences.

## Introduction 📚

In today's diverse global landscape, accessing entertainment content across different languages can be challenging. This project proposes a solution that leverages artificial intelligence to tackle these challenges head-on. The proposed model offers an efficient approach to dubbing videos, transcending traditional constraints such as lip synchronization and emotional expression.

## Features 🌟

- **Translation**: The project incorporates a sequence-to-sequence neural machine translation model with attention mechanisms for accurate translation between languages.
- **Speech Generation**: An advanced Text-to-Speech (TTS) model, such as XTTSv2, is employed to generate natural-sounding dubbed speech in the target language.
- **Lip Synchronization**: The model utilizes computer vision techniques to synchronize the talking face with the dubbed speech, ensuring precise lip movements.

## Methodology 🧮

1. **Speech Recognition**: The model uses a pre-trained Distil-whisper v3 large model for speech recognition while comparing with our small CNN-RNN speech to text model, converting the input audio to text.
2. **Translation**: A bidirectional LSTM with gradient clipping and word tokenization is used for neural machine translation, translating the text to the target language.
3. **Speech Generation**: The XTTSv2 model, combining a GPT2 backbone, Discrete VAE, Perceiver model, and HifiGAN, is employed for generating dubbed speech in the target language.
4. **Lip Synchronization**: The Wav2Lip model is utilized to synchronize the lip movements with the dubbed audio, ensuring precise lip-syncing.


## Usage 🖊️

1. Prepare the input video file with English audio.
2. Open the main ipynb file.
3. Run all, select desired option.

## Acknowledgments 🙏

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [XTTSv2](https://github.com/coqui-ai/XTTS-v2)
- [Deep Speech 2](https://arxiv.org/abs/1512.02595)

## References 📚

The references for the various models and techniques used in this project can be found in the [poster](poster.pdf).

## Outputs 




https://github.com/kmAyush/Video-Dubbing-with-Lip-Synchronization/assets/66105604/275425e0-84f8-4826-a10e-f17287d7b191




https://github.com/kmAyush/Video-Dubbing-with-Lip-Synchronization/assets/66105604/53a6ed35-c678-4643-96cf-ba187746a597



https://github.com/kmAyush/Video-Dubbing-with-Lip-Synchronization/assets/66105604/95a93f51-44b2-4d03-a1f4-a924ab768fa2




