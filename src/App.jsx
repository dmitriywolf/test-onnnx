import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

async function loadOnnxModel() {
  ort.env.wasm.numThreads = 1;

  return await ort.InferenceSession.create(
    "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
    { executionProviders: ["wasm"] }
  );
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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

      const results = await sessionRef.current.run({ images: tensor });
      // ✅ явно освобождаем память (удаляем ссылки)
      Object.keys(results).forEach((key) => {
        results[key] = null;
      });

      setCount((p) => p + 1);

      // ✅ Пауза 200 мс, чтобы GC успел очистить память
      await sleep(400);

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
