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

// Вывод JS Heap (Chrome)
const logMemory = () => {
  if (performance.memory) {
    const used = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
    console.log(`Used JS Heap: ${used} MB`);
  }
};

function App() {
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

        // Создание объекта feeds с именами входов модели
        const feeds = { a: tensorA, b: tensorB };

        // Выполнение инференса
        let results = await sessionRef.current.run(feeds);
        for (const key in results) results[key] = null;

        results = null;

        setCount((p) => p + 1);
        logMemory();

        // ⏳ Добавим мягкую задержку
        await sleep(200);
      } catch (e) {
        console.error("Inference error:", e);
      }

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

  return <p>Inference # {count}</p>;
}

export default App;
