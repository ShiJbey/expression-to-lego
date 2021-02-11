# Lego Smile Detector

By: Shi Johnson-Bey and Sabrina Fielder
Detects when a peron in the camera smiles and shows them a lego face corresposdning to ther faial expressions (only distinguishes smiles and neutral faces).

Trained on data from the [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html).

Uses TensorflowJS prebuild [facial landmark detector](https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection) model and
a model trained on feature vectors extracted from facial landmark data.

Feature vectors include:

- average inner mouth height
- average outer mouth height
- average inner mouth width
- average outer mouth width
- face_width
- face_height

![Demo output](demo.gif)

## Running the code

You will need to serve the files within the `public/` directory using either nodejs or python

NodeJS:

```bash
$ npm i -g http-server

$ http-server
```

Python:

```bash
$ python -m http.server -d public/ 8080
```

## Google Colab for Training

[https://colab.research.google.com/drive/1VeC_udgFBFWYVEyUF-vv5ZXUHlJmmZfY?usp=sharing](https://colab.research.google.com/drive/1VeC_udgFBFWYVEyUF-vv5ZXUHlJmmZfY?usp=sharing)
