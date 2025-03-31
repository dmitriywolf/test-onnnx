import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

// Загрузка модели
async function loadOnnxModel() {
  return await ort.InferenceSession.create(
    "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
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

export default function OnnxPage() {
  const sessionRef = useRef(null);
  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
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
        const tensor = new ort.Tensor(
          "float32",
          dummyDataRef.current,
          [1, 3, 320, 320]
        );

        const start = performance.now();
        let results = await sessionRef.current.run({ images: tensor });
        const end = performance.now();

        // Очистка
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
      <p>ONNX MODEL TEST</p>
      <p>Inference #: {count}</p>
      <p>Last Inference Time: {inferenceTime} ms</p>
      <p>Used JS Heap: {memoryUsage} MB</p>
    </div>
  );
}

// import { useEffect, useRef, useState } from "react";
// import * as ort from "onnxruntime-web";

// // Настройка ONNX среды
// ort.env.wasm.numThreads = 1;
// ort.env.wasm.simd = true;
// ort.env.debug = false;

// function App() {
//   const sessionRef = useRef(null);
//   const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
//   const cancelledRef = useRef(false);
//   const counterRef = useRef(0);

//   const [count, setCount] = useState(0);
//   const [memory, setMemory] = useState("0.00");
//   const [loading, setLoading] = useState(true);

//   const logMemory = () => {
//     if (performance.memory) {
//       const used = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
//       setMemory(used);
//     } else {
//       setMemory("Not available");
//     }
//   };

//   useEffect(() => {
//     cancelledRef.current = false;

//     async function loadModel() {
//       try {
//         const session = await ort.InferenceSession.create(
//           "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
//           { executionProviders: ["wasm"] }
//         );
//         sessionRef.current = session;
//         setLoading(false);
//         loop();
//       } catch (err) {
//         console.error("Failed to load model", err);
//       }
//     }

//     async function loop() {
//       while (!cancelledRef.current) {
//         let tensor = null;
//         let results = null;

//         try {
//           tensor = new ort.Tensor(
//             "float32",
//             dummyDataRef.current,
//             [1, 3, 320, 320]
//           );
//           results = await sessionRef.current.run({ images: tensor });

//           // Очистка результатов
//           for (const key in results) {
//             results[key] = null;
//           }

//           counterRef.current += 1;
//           setCount(counterRef.current);
//           logMemory();
//         } catch (e) {
//           console.error("Inference error", e);
//           break;
//         } finally {
//           // Обнуляем всё вручную
//           results = null;
//           tensor = null;

//           // Даём GC время на очистку
//           await new Promise((r) => setTimeout(r, 3000));
//         }
//       }
//     }

//     loadModel();

//     return () => {
//       cancelledRef.current = true;
//       sessionRef.current = null;
//     };
//   }, []);

//   return (
//     <div>
//       <h2>{loading ? "Loading model..." : `Inference count: ${count}`}</h2>
//       {!loading && <p>Used JS Heap: {memory} MB</p>}
//     </div>
//   );
// }

// export default App;
