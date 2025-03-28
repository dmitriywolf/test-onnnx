import { useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

async function loadOnnxModel() {
  ort.env.wasm.numThreads = 1;
 
  return await ort.InferenceSession.create(
    "https://dmitriywolf.github.io/test-onnnx/models/detector_documents_leyolo_n.onnx",
    { executionProviders: ["wasm"] }
  );
}

async function runInference(session, tensor) {
  const results = await session.run({ images: tensor });
  for (const key in results) delete results[key];
}

function App() {
  const sessionRef = useRef(null);
  const dummyDataRef = useRef(new Float32Array(1 * 3 * 320 * 320));
  const cancelledRef = useRef(false);
  const animationFrameRef = useRef(null);
  const inferenceCountRef = useRef(0);
  const lastDOMUpdateRef = useRef(0);
  const textRef = useRef(null);

  useEffect(() => {
    cancelledRef.current = false;

    async function loop() {
      if (cancelledRef.current || !sessionRef.current) return;

      const tensor = new ort.Tensor(
        "float32",
        dummyDataRef.current,
        [1, 3, 320, 320]
      );

      await runInference(sessionRef.current, tensor);

      inferenceCountRef.current += 1;

      const now = performance.now();
      // DOM обновляется максимум раз в секунду
      if (now - lastDOMUpdateRef.current > 1000) {
        textRef.current.textContent = `Inferences: ${inferenceCountRef.current}`;
        lastDOMUpdateRef.current = now;
      }

      animationFrameRef.current = requestAnimationFrame(loop);
    }

    async function init() {
      textRef.current.textContent = "Loading...";
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

  return <p ref={textRef}>Loading...</p>;
}

export default App;
