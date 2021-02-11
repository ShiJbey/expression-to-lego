// import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs';
import { interval } from 'rxjs';
import { tap } from 'rxjs';
// import '@tensorflow/tfjs-converter';
// import '@tensorflow/tfjs-backend-wasm';

const PREDICTION_STR = ['NEUTRAL', 'SMILING'];
const QUESTION_IMG = document.getElementById('question-img');
const HAPPY_IMG = document.getElementById('happy-img');
const NEUTRAL_IMG = document.getElementById('neutral-img');
const PREDICTION_TEXT = document.getElementById('prediction');

function getFeatureVector(facePrediction) {
  const lip_lower_inner = tf.tensor2d(facePrediction.annotations.lipsLowerInner);
  const lip_upper_inner = tf.tensor2d(facePrediction.annotations.lipsUpperInner);
  const lip_lower_outer = tf.tensor2d(facePrediction.annotations.lipsLowerOuter);
  const lip_upper_outer = tf.tensor2d(facePrediction.annotations.lipsUpperOuter);
  const silhouette = tf.tensor2d(facePrediction.annotations.silhouette);

  const avg_height_inner = tf.abs(
    tf.sub(
      tf.mean(lip_upper_inner, 0),
      tf.mean(lip_lower_inner, 0)
    )).dataSync()[1];

  const avg_height_outer = tf.abs(
    tf.sub(
      tf.mean(lip_upper_inner, 0),
      tf.mean(lip_lower_inner, 0)
    )).dataSync()[1]

  const width_upper_inner = tf.abs(
    tf.sub(
      tf.max(lip_upper_inner, 0),
      tf.min(lip_upper_inner, 0)
    )).dataSync()[0];

  const width_lower_inner = tf.abs(
    tf.sub(
      tf.max(lip_lower_inner, 0),
      tf.min(lip_lower_inner, 0)
    )).dataSync()[0];

  const width_upper_outer = tf.abs(
    tf.sub(
      tf.max(lip_upper_outer, 0),
      tf.min(lip_upper_outer, 0)
    )).dataSync()[0];

  const width_lower_outer = tf.abs(
    tf.sub(
      tf.max(lip_lower_outer, 0),
      tf.min(lip_lower_outer, 0)
    )).dataSync()[0];

  const avg_width_inner = (width_lower_inner + width_upper_inner) / 2.0
  const avg_width_outer = (width_lower_outer + width_upper_outer) / 2.0

  const [face_width, face_height] = tf.abs(
    tf.sub(
      tf.max(silhouette, 0),
      tf.min(silhouette, 0)
    )).dataSync().slice(0, 2);

  const featureVect = tf.tensor([[
    avg_height_inner,
    avg_height_outer,
    avg_width_inner,
    avg_width_outer,
    face_width,
    face_height
  ]]);

  return featureVect;
}


async function getMedia(contraints) {
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia(contraints);
    const videoHTML = document.getElementById('webcam-video');
    videoHTML.srcObject = stream;
    videoHTML.play();
  } catch (err) {
    console.error(err);
  }
}

function showQuestionMark() {
  QUESTION_IMG.style.display = 'block';
  HAPPY_IMG.style.display = 'none';
  NEUTRAL_IMG.style.display = 'none';
}

function showHappyImg(imgPath) {
  if (imgPath) {
    HAPPY_IMG.src = imgPath;
  }
  QUESTION_IMG.style.display = 'none';
  HAPPY_IMG.style.display = 'block';
  NEUTRAL_IMG.style.display = 'none';
}

function showNeutralImg(imgPath) {
  if (imgPath){
    NEUTRAL_IMG.SRC = imgPath;
  }
  QUESTION_IMG.style.display = 'none';
  HAPPY_IMG.style.display = 'none';
  NEUTRAL_IMG.style.display = 'block';
}

(async function main() {
  await tf.ready();

  let videoReady = false;

  const smileModel = await tf.loadLayersModel('http://localhost:8080/model/model.json');

  // Load the MediaPipe Facemesh package.
  const model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);

  const videoHTML = document.getElementById('webcam-video');

  videoHTML.addEventListener('loadeddata', () => {videoReady = true;});

  getMedia({
    video: true,
    audio: false
  });

  interval(300).subscribe((x) => {
    if (videoReady) {
      estimateFace().catch(console.error);
    }
  });

  async function estimateFace() {
    try {
      const [prediction] = await model.estimateFaces({
        input: videoHTML
      });

      if (prediction) {
        const vect = getFeatureVector(prediction);
        const pred = tf.reshape(smileModel.predict(vect), [2]);
        const [smilePred] = tf.argMax(pred).dataSync();
        PREDICTION_TEXT.innerHTML = PREDICTION_STR[smilePred];
        if (smilePred === 0) {
          // Neutral
          showNeutralImg();
        } else {
          showHappyImg();
        }

      } else {
        showQuestionMark();
        PREDICTION_TEXT.innerHTML = `Oh No! I can't see your face.`;
      }

    } catch(err) {
      console.error(err);
    }
  }
})();
