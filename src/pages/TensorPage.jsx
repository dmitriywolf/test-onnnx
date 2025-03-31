import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

// Загрузка модели
async function loadTensorModel() {
  try {
    await tf.ready();
    const model = await tf.loadGraphModel(
      `https://dmitriywolf.github.io/test-onnnx/models/model.json`
    );
    return {
      net: model,
      inputShape: model.inputs[0].shape,
    };
  } catch (err) {
    console.error("[error.load.model]", err);
  }
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

export default function TensorPage() {
  const modelRef = useRef(null);
  const cancelledRef = useRef(false);
  const animationFrameRef = useRef(null);

  const [count, setCount] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);

  useEffect(() => {
    cancelledRef.current = false;

    async function loop() {
      if (cancelledRef.current || !modelRef.current) return;

      try {
        tf.engine().startScope();

        const dummyInput = tf.ones(modelRef.current.inputShape);

        const start = performance.now();
        const res = await modelRef.current.net.execute(dummyInput);
        const end = performance.now();

        const timeTaken = +(end - start).toFixed(2);
        const memUsed = getMemoryUsage();

        tf.dispose([res]);
        setCount((prev) => prev + 1);
        setInferenceTime(timeTaken);
        if (memUsed !== null) setMemoryUsage(memUsed);
      } catch (e) {
        console.error("Inference error:", e);
      }

      await sleep(200); // небольшая задержка
      animationFrameRef.current = requestAnimationFrame(loop);
    }

    async function init() {
      modelRef.current = await loadTensorModel();
      if (!cancelledRef.current && modelRef.current) {
        loop();
      }
    }

    init();

    return () => {
      cancelledRef.current = true;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      modelRef.current = null;
    };
  }, []);

  return (
    <div>
      <p>TENSOR MODEL TEST</p>
      <p>Inference #: {count}</p>
      <p>Last Inference Time: {inferenceTime} ms</p>
      <p>Used JS Heap: {memoryUsage} MB</p>
    </div>
  );
}
