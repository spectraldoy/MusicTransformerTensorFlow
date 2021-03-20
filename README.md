# MusicTransformerTensorFlow
The Music Transformer is a deep learning model that builds on the Transformer by considering the relative distances between different elements of a sequence rather than simply their absolute positions in the sequence. I explored my interest in AI-generated music through this project and learned quite a bit about current research in the field of AI in terms of both algorithms and math. This repository contains a copy of the Jupyter notebook where I built and trained a Music Transformer on the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) using TensorFlow with a TPU following the description in the [Music Transformer Paper](https://arxiv.org/pdf/1809.04281.pdf), `Music_Transformer_Public.ipynb`, as well as functionality for a trained model in both PyTorch and TensorFlow that can be run on a GPU or CPU, and some generated samples. The HDF5 of my TensorFlow Music Transformer's trained weights could not be included as it exceeded 25 MB, however, a small PyTorch model is included in this repository. A few .wav samples generated using my TensorFlow model can be found in the [samples](https://github.com/spectraldoy/Music-Transformer/tree/master/samples) folder.

Refer to `Music_Transformer_Public.ipynb` for details on building and training the model with a TPU on Colab as well as notes on how the Relative Attention mechanism works. I preprocessed the data locally and uploaded these to a private Google Drive folder, which I then accessed from Colab to train my model. In order to be able to use the notebook to build and train their own model, the reader will presumably have to mimic such a setup.

TODO: use gsutil font

## Key Dependencies
1. TensorFlow 2.3.0
2. NumPy 1.19.0
3. mido 1.2.9
5. midi2audio 0.1.1

## Running from the Command Line

## Generation with Python
### Setup
To set up generation with a pretrained model (preferably on a GPU), run the following:
```python
from music_transformer import *
from tensorflow import random
transformer = TransformerDecoder(*hparams) # create the model
_ = transformer(random.uniform(2, MAX_LENGTH)) # build the model
del _

# if you have them
transformer.load_weights("model_weights.h5") 
```
Run the above in the directory where `music_transformer.py`, `transformerutil6.py` and `model_weights.h5` are all kept.

### Generate Music!
From a Python or IPython console, or a Jupyter Notebook, run the following to generate a .midi and a .wav file with a loaded/pretrained Music Transformer:
```python
generate(transformer, inp=['<start>'], path="your/path.midi", temperature=tau,
         k=top_k_samples, tempo=tempo_in_µs_per_beat, 
         wav=True, verbose=False)
```
The `inp` parameter can be any list of events from the event vocabulary specified in `transformerutil6.py`; the model will continue it. The `temperature` and `k` parameters determine the randomness of the decode sampling, the process by which the model generates every subsequent event (note that functions may also be supplied to these parameters). `k ≈ 40` and `temperature ≈ 0.9` work well for me. Set the `wav` flag to `False` (default `True`) to generate only a .midi file, and set the `verbose` flag to `True` (default `False`) to see a small set of details regarding the generation process. The `generate` function will save the generated .midi file at `your/path.midi` and will save its corresponding .wav file at `your/path.wav`. Additional methods to perform the decode sampling are `'categorical'` and `'argmax'` which you can specify with the `mode` keyword argument:
```python
generate(transformer, ['<start>'], path="your/path.midi", mode='categorical', 
         temperature=tau, tempo=tempo_in_µs_per_beat)
```
```python
generate(transformer, ['<start>'], path="your/path.midi", mode='argmax', 
         tempo=tempo_in_µs_per_beat)
```
