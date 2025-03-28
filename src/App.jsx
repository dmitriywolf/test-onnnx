// import { useEffect, useState } from "react";

// function App() {
//   const [count, setCount] = useState(0);

//   useEffect(() => {
//     // ✅ правильный способ в Vite
//     const worker = new Worker(new URL("./onnxWorker.js", import.meta.url), {
//       type: "module",
//     });

//     worker.onmessage = () => {
//       setCount((prev) => prev + 1);
//     };

//     return () => worker.terminate();
//   }, []);

//   return <p>Inference # {count}</p>;
// }

// export default App;

import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

async function loadOnnxModel() {
  ort.env.wasm.numThreads = 1;

  return await ort.InferenceSession.create(
    "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
    { executionProviders: ["webgl", "wasm"] }
  );
}

function App() {
  const sessionRef = useRef(null);

  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));

  const cancelledRef = useRef(false);
  const animationFrameRef = useRef(null);

  const [count, setCount] = useState(0);

  useEffect(() => {
    cancelledRef.current = false;

    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      const tensor = new ort.Tensor(
        "float32",
        dummyDataRef.current,
        [1, 3, 320, 320]
      );

      await sessionRef.current.run({ images: tensor });

      setCount((p) => p + 1);

      animationFrameRef.current = requestAnimationFrame(loop);
    }

    async function init() {
      sessionRef.current = await loadOnnxModel();

      if (!sessionRef.current || cancelledRef.current) return;

      loop();
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
