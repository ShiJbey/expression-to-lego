import fs from 'fs';
import faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import tf from '@tensorflow/tfjs-node';
import '@tensorflow/tfjs-backend-wasm';
tf.setBackend('tensorflow').then(main);
//
// require('@tensorflow/tfjs-core');
// require('@tensorflow/tfjs-node');

async function main() {
  try {
    const imageDir = 'data/images/';
    const landmarksDir = 'data/landmarks/';

    const model = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);

    const files = fs.readdirSync(imageDir);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      // Load and decode image file
      const [filename] = file.split('.');
      const dest = landmarksDir + filename + '.json';
      const src = imageDir + file;
      const imageBuffer = fs.readFileSync(src);
      const imageTensor = tf.node.decodeImage(imageBuffer);

      // Take the first face
      const [prediction] = await model.estimateFaces({
        input: imageTensor
      });

      if (prediction) {
        fs.writeFileSync(dest, JSON.stringify(prediction.annotations));
        console.log(dest);
      }
    }

  } catch (err) {
    console.error(err);
  }
};
