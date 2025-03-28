// onnxWorker.js
import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.proxy = true;

let session = null;
const dummyData = new Float32Array(1 * 3 * 320 * 320);

async function init() {
  session = await ort.InferenceSession.create(
    "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
    { executionProviders: ["wasm"] }
  );
  loop();
}

async function loop() {
  const tensor = new ort.Tensor("float32", dummyData, [1, 3, 320, 320]);
  const results = await session.run({ images: tensor });

  postMessage({ done: true });

  for (const key in results) results[key] = null;

  requestAnimationFrame(loop);
}

init();
