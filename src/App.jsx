import { Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import TensorPage from "./pages/TensorPage";
import OnnxPage from "./pages/OnnxPage";
import OnnxLightPage from "./pages/OnnxLightPage";

function App() {
  return (
    <div>
      <nav style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <Link to="/">Home</Link>
        <Link to="/tensor">TensorFlow</Link>
        <Link to="/onnx">ONNX</Link>
        <Link to="/onnx-light">ONNX LIGHT FROM DOC</Link>
      </nav>

      <Routes>
        <Route path="/test-onnnx/" element={<Home />} />
        <Route path="/test-onnnx/tensor" element={<TensorPage />} />
        <Route path="/test-onnnx/onnx" element={<OnnxPage />} />
        <Route path="/test-onnnx/onnx-light" element={<OnnxLightPage />} />
      </Routes>
    </div>
  );
}

export default App;
