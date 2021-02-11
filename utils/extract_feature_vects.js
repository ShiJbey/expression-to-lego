import fs from 'fs';

(function main() {
  const landmarksFile = 'data/landmarks.json';

  try {
    const landmarkData = JSON.parse(fs.readFileSync(landmarksFile));

    console.log(landmarkData);
  } catch (err) {
    console.error(err);
  }
})();
