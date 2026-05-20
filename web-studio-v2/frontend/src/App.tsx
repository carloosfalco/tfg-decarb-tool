import { Navigate, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Tool from "./pages/Tool";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/tool" element={<Tool />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
