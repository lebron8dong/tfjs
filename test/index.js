import {MnistData} from './data.js';

async function doPrediction(model, data, testDataSize = 3) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels;
  const preds = model.predict(testxs);
  console.log(preds.data());//error:Uncaught TypeError: tf.util.convertBackendValuesAndArrayBuffer is not a function
  
  testxs.dispose();
  labels.dispose();
  preds.dispose();
}

await tf.setBackend('webgpu');

const data = new MnistData();
await data.load();

const model = await tf.loadLayersModel('./tfmodel.json');
await doPrediction(model,data);
