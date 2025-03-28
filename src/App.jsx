import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// Настройка ONNX среды
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.debug = false;

function App() {
  const sessionRef = useRef(null);
  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
  const cancelledRef = useRef(false);
  const counterRef = useRef(0);

  const [count, setCount] = useState(0);
  const [memory, setMemory] = useState("0.00");
  const [loading, setLoading] = useState(true);

  // Показ использования памяти (только в Chrome)
  const logMemory = () => {
    if (performance.memory) {
      const used = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
      setMemory(used);
    }
  };

  useEffect(() => {
    cancelledRef.current = false;

    async function loadModel() {
      try {
        const session = await ort.InferenceSession.create(
          "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
          { executionProviders: ["wasm"] }
        );
        sessionRef.current = session;
        setLoading(false);
        loop();
      } catch (err) {
        console.error("Failed to load model", err);
      }
    }

    async function loop() {
      while (!cancelledRef.current) {
        const tensor = new ort.Tensor(
          "float32",
          dummyDataRef.current,
          [1, 3, 320, 320]
        );

        try {
          const results = await sessionRef.current.run({ images: tensor });
          Object.keys(results).forEach((k) => (results[k] = null));

          counterRef.current += 1;
          setCount(counterRef.current);
          logMemory();
        } catch (e) {
          console.error("Inference error", e);
          break;
        }

        await new Promise((r) => setTimeout(r, 400));
      }
    }

    loadModel();

    return () => {
      cancelledRef.current = true;
      sessionRef.current = null;
    };
  }, []);

  return (
    <div>
      <h2>{loading ? "Loading model..." : `Inference count: ${count}`}</h2>
      {!loading && <p>Used JS Heap: {memory} MB</p>}
    </div>
  );
}

export default App;

// import { useState, useEffect, useRef } from "react";
// import * as ort from "onnxruntime-web";

// async function loadOnnxModel() {
//   return await ort.InferenceSession.create(
//     "https://dmitriywolf.github.io/test-onnnx/models/best_uint8.onnx",
//     // "/models/best_uint8.onnx",
//     { executionProviders: ["wasm"] }
//   );
// }
// const logMemory = () => {
//   if (performance.memory) {
//     const used = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
//     console.log(`Used JS Heap: ${used} MB`);
//   }
// };
// function App() {
//   const sessionRef = useRef(null);

//   const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));

//   const cancelledRef = useRef(false);
//   const animationFrameRef = useRef(null);

//   const [count, setCount] = useState(0);

//   useEffect(() => {
//     cancelledRef.current = false;

//     async function loop() {
//       if (cancelledRef.current || !sessionRef.current) return;

//       const tensor = new ort.Tensor(
//         "float32",
//         dummyDataRef.current,
//         [1, 3, 320, 320]
//       );

//       let results = await sessionRef.current.run({ images: tensor });

//       // eslint-disable-next-line no-unused-vars
//       results = null;

//       setCount((p) => p + 1);
//       logMemory()

//       animationFrameRef.current = requestAnimationFrame(loop);
//     }

//     async function init() {
//       sessionRef.current = await loadOnnxModel();

//       if (!sessionRef.current || cancelledRef.current) return;

//       loop();
//     }

//     init();

//     return () => {
//       cancelledRef.current = true;
//       if (animationFrameRef.current)
//         cancelAnimationFrame(animationFrameRef.current);
//       sessionRef.current = null;
//     };
//   }, []);

//   return <p>Inference # {count}</p>;
// }

// export default App;
