/**
 * Filters out images from the FEI Face Database
 * so that we only have left profile, front smile,
 * front neutral, and right profile
 */
const fs = require('fs');

(function main() {

  const inDir = 'expression_images/';
  const outDir = 'data/';

  const labelMap = {
    '01': 'LEFT',
    '10': 'RIGHT',
    '11': 'NEUTRAL',
    '12': 'SMILING'
  };

  try {
    const files = fs.readdirSync(inDir);

    files.forEach((file) => {
      // Copy each file and replace the second number
      // with the matching label
      const [filename, extension] = file.split('.');
      const [sample, imageType] = filename.split('-');

      if (Object.keys(labelMap).includes(imageType)) {
        const dest = `${sample}_${labelMap[imageType]}.${extension}`;
        fs.copyFileSync(inDir + file, outDir + dest);
        console.log(outDir + dest);
      }
    });
  } catch (err) {
    console.error(err);
  }
})();
