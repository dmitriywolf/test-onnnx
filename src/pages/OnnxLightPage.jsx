import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

// Загрузка модели
async function loadOnnxModel() {
  return await ort.InferenceSession.create(
    "https://dmitriywolf.github.io/test-onnnx/models/model.onnx",
    { executionProviders: ["wasm"] }
  );
}

// Пауза через промис
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Получение использования JS Heap (в MB)
const getMemoryUsage = () => {
  if (performance.memory) {
    return +(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
  }
  return null;
};

export default function OnnxLightPage() {
  const sessionRef = useRef(null);
  const dummyDataARef = useRef(
    Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  );
  const dummyDataBRef = useRef(
    Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  );
  const cancelledRef = useRef(false);
  const animationFrameRef = useRef(null);

  const [count, setCount] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);

  useEffect(() => {
    cancelledRef.current = false;

    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      try {
        const tensorA = new ort.Tensor(
          "float32",
          dummyDataARef.current,
          [3, 4]
        );
        const tensorB = new ort.Tensor(
          "float32",
          dummyDataBRef.current,
          [4, 3]
        );

        const feeds = { a: tensorA, b: tensorB };

        const start = performance.now();
        let results = await sessionRef.current.run(feeds);
        const end = performance.now();

        for (const key in results) results[key] = null;
        results = null;

        const timeTaken = +(end - start).toFixed(2);
        const memUsed = getMemoryUsage();

        setCount((p) => p + 1);
        setInferenceTime(timeTaken);
        if (memUsed !== null) setMemoryUsage(memUsed);
      } catch (e) {
        console.error("Inference error:", e);
      }

      await sleep(200);
      animationFrameRef.current = requestAnimationFrame(loop);
    }

    async function init() {
      sessionRef.current = await loadOnnxModel();
      if (!cancelledRef.current) loop();
    }

    init();

    return () => {
      cancelledRef.current = true;
      if (animationFrameRef.current)
        cancelAnimationFrame(animationFrameRef.current);
      sessionRef.current = null;
    };
  }, []);

  return (
    <div>
      <p>ONNX LIGHT MODEL TEST</p>
      <p>Inference #: {count}</p>
      <p>Last Inference Time: {inferenceTime} ms</p>
      <p>Used JS Heap: {memoryUsage} MB</p>
    </div>
  );
}
