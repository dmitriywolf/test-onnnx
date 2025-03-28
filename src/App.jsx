import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

async function loadOnnxModel() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = true;

  try {
    return await ort.InferenceSession.create(
      "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
      { executionProviders: ["wasm"] }
    );
  } catch (err) {
    console.error("[ONNX] Ошибка загрузки модели:", err);
    return null;
  }
}

async function runInference(session, tensor) {
  try {
    const results = await session.run({ images: tensor });
    for (const key in results) delete results[key];
  } catch (e) {
    console.error("Ошибка инференса:", e);
  }
}

function App() {
  const animationFrameRef = useRef(null);
  const sessionRef = useRef(null);
  const cancelledRef = useRef(false);
  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
  const lastInferenceTimeRef = useRef(0);
  const inferenceInterval = 200; // 5 fps

  // возвращаем useState для управления DOM
  const [status, setStatus] = useState({ loading: true, count: 0, time: 0 });

  useEffect(() => {
    cancelledRef.current = false;

    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      const now = performance.now();
      if (now - lastInferenceTimeRef.current > inferenceInterval) {
        lastInferenceTimeRef.current = now;

        const tensor = new ort.Tensor(
          "float32",
          dummyDataRef.current,
          [1, 3, 320, 320]
        );

        const t0 = performance.now();
        await runInference(sessionRef.current, tensor);
        const t1 = performance.now();

        // обновляем состояние только один раз за инференс
        setStatus((prev) => ({
          loading: false,
          count: prev.count + 1,
          time: t1 - t0,
        }));
      }

      animationFrameRef.current = requestAnimationFrame(loop);
    }

    async function init() {
      setStatus({ loading: true, count: 0, time: 0 });

      const session = await loadOnnxModel();
      if (!session || cancelledRef.current) return;

      sessionRef.current = session;

      // пауза перед стартом (Safari iOS)
      setTimeout(() => {
        if (!cancelledRef.current) loop();
      }, 300);
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
    <p>
      {status.loading
        ? "Loading..."
        : `Inference №${status.count} | Time: ${status.time.toFixed(2)} ms`}
    </p>
  );
}

export default App;
