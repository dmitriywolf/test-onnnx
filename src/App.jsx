import { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";

// Загрузка ONNX-модели
async function loadOnnxModel() {
  try {
    ort.env.wasm.numThreads = 1;

    const session = await ort.InferenceSession.create(
      "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
      {
        executionProviders: ["wasm"],
      }
    );

    console.log("[ONNX] Модель загружена");
    return session;
  } catch (err) {
    console.error("[ONNX] Ошибка загрузки модели:", err);
    return null;
  }
}

// Инференс для одного кадра
async function runInference(session, tensor, setCount) {
  try {
    const t0 = performance.now();
    const results = await session.run({ images: tensor });
    const t1 = performance.now();

    setCount((prev) => ({ c: prev.c + 1, t: t1 - t0 }));

    for (const key in results) {
      delete results[key];
    }
  } catch (e) {
    console.error("❌ Ошибка инференса:", e);
  }
}

function App() {
  const [count, setCount] = useState({ c: 0, t: 0 });
  const [isLoading, setIsLoading] = useState(true); // 🔹 индикатор загрузки

  const sessionRef = useRef(null);
  const animationFrameRef = useRef(null);
  const isInferenceRunningRef = useRef(false);
  const lastInferenceTimeRef = useRef(0);
  const cancelledRef = useRef(false);
  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
  const inferenceInterval = 100;

  useEffect(() => {
    cancelledRef.current = false;

    async function init() {
      const session = await loadOnnxModel();
      if (!session || cancelledRef.current) return;

      sessionRef.current = session;
      setIsLoading(false); // 🔹 модель загружена — скрываем индикатор
      loop();
    }

    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      const now = performance.now();
      const enoughTimePassed =
        now - lastInferenceTimeRef.current > inferenceInterval;

      if (!isInferenceRunningRef.current && enoughTimePassed) {
        isInferenceRunningRef.current = true;
        lastInferenceTimeRef.current = now;

        const tensor = new ort.Tensor(
          "float32",
          dummyDataRef.current,
          [1, 3, 320, 320]
        );

        await runInference(sessionRef.current, tensor, setCount);
        isInferenceRunningRef.current = false;
      }

      animationFrameRef.current = requestAnimationFrame(loop);
    }

    init();

    return () => {
      cancelledRef.current = true;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      sessionRef.current = null;
    };
  }, []);

  return isLoading ? (
    <p>Loading...</p> // 🔹 пока модель грузится
  ) : (
    <p>
      Inference №{count.c} | Time: {count.t.toFixed(2)} ms
    </p>
  );
}

export default App;
