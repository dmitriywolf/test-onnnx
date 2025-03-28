import { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";

function App() {
  const [count, setCount] = useState(0);
  const animationFrameRef = useRef(null);
  // Флаг, указывающий на то, что инференс выполняется
  const isInferenceRunningRef = useRef(false);

  useEffect(() => {
    let session = null;

    async function loadModelAndStartAnimation() {
      try {
        // Загрузка модели
        session = await ort.InferenceSession.create("https://github.com/dmitriywolf/test-onnnx/models/model.onnx");

        // Функция, выполняющая инференс на каждом кадре
        async function runFrame() {
          // Если инференс уже выполняется, пропускаем этот кадр
          if (isInferenceRunningRef.current) {
            animationFrameRef.current = requestAnimationFrame(runFrame);
            return;
          }
          isInferenceRunningRef.current = true;

          // Подготовка входных данных
          const dataA = Float32Array.from([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          ]);
          const dataB = Float32Array.from([
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
          ]);
          const tensorA = new ort.Tensor("float32", dataA, [3, 4]);
          const tensorB = new ort.Tensor("float32", dataB, [4, 3]);

          // Создание объекта feeds с именами входов модели
          const feeds = { a: tensorA, b: tensorB };

          try {
            // Выполнение инференса
            const results = await session.run(feeds);
            const dataC = results.c.data;
            console.log("DATA C", dataC);
          } catch (e) {
            console.error("Ошибка инференса:", e);
          } finally {
            // Сбрасываем флаг по окончании инференса
            isInferenceRunningRef.current = false;
          }

          // Обновление счетчика для отображения количества выполненных инференсов
          setCount((prev) => prev + 1);

          // Планируем выполнение на следующем кадре
          animationFrameRef.current = requestAnimationFrame(runFrame);
        }

        runFrame();
      } catch (err) {
        console.error("[ONNX] Ошибка загрузки модели:", err);
      }
    }

    loadModelAndStartAnimation();

    // Очистка эффекта: отменяем планируемый кадр при размонтировании компонента
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return <p>Inference N {count}</p>;
}

export default App;
