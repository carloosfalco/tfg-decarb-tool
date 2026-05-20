import { Component, type ErrorInfo, type PropsWithChildren, type ReactNode } from "react";

type ErrorBoundaryState = {
  error: Error | null;
};

export class ErrorBoundary extends Component<PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Unhandled UI error", error, errorInfo);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;

    return (
      <main className="app-fallback">
        <section className="panel app-fallback__panel">
          <span className="badge">Error recuperable</span>
          <h1>La interfaz se ha protegido ante un fallo inesperado</h1>
          <p>
            Puedes recargar la página y continuar. Si vuelve a ocurrir, revisa que el backend esté activo y que los
            datos introducidos tengan formato válido.
          </p>
          <button className="button button--primary" type="button" onClick={() => window.location.reload()}>
            Recargar
          </button>
        </section>
      </main>
    );
  }
}
