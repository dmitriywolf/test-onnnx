import { useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

async function loadOnnxModel() {
  ort.env.wasm.numThreads = 1;

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
  const isInferenceRunningRef = useRef(false);
  const lastInferenceTimeRef = useRef(0);
  const cancelledRef = useRef(false);
  const inferenceInterval = 200;
  const inferenceCountRef = useRef(0);
  const inferenceTimeRef = useRef(0);

  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
  const textRef = useRef(null);

  useEffect(() => {
    cancelledRef.current = false;

    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      const now = performance.now();
      if (
        !isInferenceRunningRef.current &&
        now - lastInferenceTimeRef.current > inferenceInterval
      ) {
        isInferenceRunningRef.current = true;
        lastInferenceTimeRef.current = now;

        const tensor = new ort.Tensor(
          "float32",
          dummyDataRef.current,
          [1, 3, 320, 320]
        );

        const t0 = performance.now();
        await runInference(sessionRef.current, tensor);
        const t1 = performance.now();

        inferenceCountRef.current += 1;
        inferenceTimeRef.current = t1 - t0;

        textRef.current.textContent = `Inference №${
          inferenceCountRef.current
        } | Time: ${inferenceTimeRef.current.toFixed(2)} ms`;

        isInferenceRunningRef.current = false;
      }

      animationFrameRef.current = requestAnimationFrame(loop);
    }

    async function init() {
      textRef.current.textContent = "Loading...";
      const session = await loadOnnxModel();

      if (!session || cancelledRef.current) return;

      sessionRef.current = session;

      setTimeout(() => {
        if (cancelledRef.current) return;
        loop();
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

  return <p ref={textRef}>Loading...</p>;
}

export default App;
