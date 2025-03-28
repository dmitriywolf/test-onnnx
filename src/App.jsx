import { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";

// Загрузка модели
async function loadOnnxModel() {
  try {
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

// Один кадр инференса
async function runInference(session, setCount) {
  const dummyData = new Float32Array(1 * 3 * 320 * 320);
  for (let i = 0; i < dummyData.length; i++) {
    dummyData[i] = (i % 256) / 255;
  }

  const inputTensor = new ort.Tensor("float32", dummyData, [1, 3, 320, 320]);

  try {
    const t0 = performance.now();
    const results = await session.run({ images: inputTensor });
    const t1 = performance.now();

    setCount((prev) => ({ c: prev.c + 1, t: t1 - t0 }));

    // Очистка результатов
    for (const key in results) {
      delete results[key];
    }
  } catch (e) {
    console.error("❌ Ошибка инференса:", e);
  }
}

// Главный компонент
function App() {
  const [count, setCount] = useState({ c: 0, t: 0 });
  const animationFrameRef = useRef(null);
  const isInferenceRunningRef = useRef(false);
  const sessionRef = useRef(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;

    async function init() {
      const session = await loadOnnxModel();
      if (session) {
        sessionRef.current = session;
        loop(); // стартуем цикл инференса
      }
    }

    // Цикл инференса по кадрам
    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      if (!isInferenceRunningRef.current) {
        isInferenceRunningRef.current = true;

        await runInference(sessionRef.current, setCount);

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

  return (
    <p>
      Inference №{count.c} | Time: {count.t.toFixed(2)} ms
    </p>
  );
}

export default App;
