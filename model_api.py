
import os
from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
import librosa
import numpy as np

app = Flask(__name)

# Define the path to your ONNX model
ONNX_MODEL_PATH = "onnx_models/discogs-effnet-bsdynamic-1.onnx"

# Initialize the ONNX model
onnx_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)


#model = onnx.load(ONNX_MODEL_PATH)
#input_name = model.graph.input[0].name


FFT_HOP = 256
FFT_SIZE = 512
N_MELS = 96
input_length = 2.05
SR = 16000
n_frames = librosa.time_to_frames(input_length, sr=SR, n_fft=FFT_SIZE, hop_length=FFT_HOP) + 1

def spectro2batch(song_file:str) -> np.ndarray:
    signal,_ = librosa.load(song_file, sr = SR)
    audio_rep = librosa.feature.melspectrogram(y=signal,sr=SR,hop_length=FFT_HOP,n_fft=FFT_SIZE, n_mels=N_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)
    last_frame = audio_rep.shape[0] - n_frames + 1
    first = True
    for time_stamp in range(0, last_frame, n_frames):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)
    return batch

#def _inference( batch:np.ndarray) -> dict:
#   outputs = onnx_session.run(None, {input_name: batch},)  
#   o_names = [i.name for i in self.model.graph.output]
#   outputs = dict(zip(o_names, outputs))
#   return outputs





# Define the endpoint for audio prediction
@app.route('/predict', methods=['POST'])
def predict_audio():
    try:
        # Ensure the request contains audio data (e.g., in WAV format)
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio data provided'})

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Perform any necessary pre-processing on the audio data
        # For example, convert the audio file to the required input format
        # You should adjust this code to match your model's input requirements

        # Read audio data and convert to the required format (e.g., float32)
        audio_data = spectro2batch(audio_file)

        # Run inference on the ONNX model
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        input_data = {input_name: audio_data}
        prediction = onnx_session.run([output_name], input_data)[0]

        # Perform post-processing on the prediction if needed

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

#def process_audio(audio_file):
#   # Perform any necessary audio processing here, such as reading the file,
#   # converting it to the required format, and reshaping it as needed.
#   # You should adapt this function based on your model's requirements.
#   return np.random.rand(1, 1, 128, 128).astype(np.float32)  # Placeholder for demonstration

if __name__ == '__main__':
    app.run(debug=True)