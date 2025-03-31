import { useState } from "react";
import Home from "./pages/Home";
import TensorPage from "./pages/TensorPage";
import OnnxPage from "./pages/OnnxPage";
import OnnxLightPage from "./pages/OnnxLightPage";

function App() {
  const [activeTab, setActiveTab] = useState("home");

  const renderTab = () => {
    switch (activeTab) {
      case "home":
        return <Home />;
      case "tensor":
        return <TensorPage />;
      case "onnx":
        return <OnnxPage />;
      case "onnx-light":
        return <OnnxLightPage />;
      default:
        return <Home />;
    }
  };

  return (
    <div>
      <nav style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <button onClick={() => setActiveTab("home")}>Home</button>
        <button onClick={() => setActiveTab("tensor")}>TensorFlow</button>
        <button onClick={() => setActiveTab("onnx")}>ONNX</button>
        <button onClick={() => setActiveTab("onnx-light")}>
          ONNX LIGHT FROM DOC
        </button>
      </nav>

      <div>{renderTab()}</div>
    </div>
  );
}

export default App;
