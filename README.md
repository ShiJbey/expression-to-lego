# Lego Smile Detector

Detects when a peron in the camera smiles and shows them a lego face corresposdning to ther faial expressions (only distinguishes smiles and neutral faces).

Trained on data from the [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html).

Uses TensorflowJS prebuild [facial landmark detector](https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection) model and
a model trained on feature vectors extracted from facial landmark data.

## Google Colab for Training

[https://colab.research.google.com/drive/1VeC_udgFBFWYVEyUF-vv5ZXUHlJmmZfY?usp=sharing](https://colab.research.google.com/drive/1VeC_udgFBFWYVEyUF-vv5ZXUHlJmmZfY?usp=sharing)
