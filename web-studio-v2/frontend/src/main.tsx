import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ErrorBoundary } from "./components/system/ErrorBoundary";
import { AppStateProvider } from "./store/AppState";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppStateProvider>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </AppStateProvider>
    </BrowserRouter>
  </React.StrictMode>
);
