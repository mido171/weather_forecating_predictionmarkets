import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import AppRuntimeBoundary from "./AppRuntimeBoundary.jsx";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <AppRuntimeBoundary>
    <App />
  </AppRuntimeBoundary>
);
