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