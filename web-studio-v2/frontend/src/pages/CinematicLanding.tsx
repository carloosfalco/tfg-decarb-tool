import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";

const floatingCards = [
  {
    title: "Calculo de huella de carbono",
    detail: "Alcance 1 y 2 con trazabilidad metodologica.",
    metric: "tCO2e",
  },
  {
    title: "Analisis del contexto mediante IA",
    detail: "Lectura PESTEL para decisiones industriales.",
    metric: "PESTEL",
  },
  {
    title: "Generacion de iniciativas",
    detail: "Medidas de reduccion adaptadas al perfil operativo.",
    metric: "IA",
  },
  {
    title: "Evaluacion economica y ambiental",
    detail: "Impacto, coste, ahorro y retorno esperado.",
    metric: "ROI",
  },
  {
    title: "Optimizacion del portafolio",
    detail: "Priorizacion bajo presupuesto y objetivos de reduccion.",
    metric: "OPT",
  },
];

const steps = [
  "Introduccion de datos de consumo y actividad.",
  "Calculo de emisiones de Alcance 1 y Alcance 2.",
  "Analisis PESTEL asistido por IA.",
  "Generacion y evaluacion de iniciativas.",
  "Optimizacion del portafolio bajo restricciones.",
];

const signals = [
  { label: "Emisiones estimadas", value: "1.842 tCO2e" },
  { label: "Reduccion potencial", value: "-28%" },
  { label: "Portfolio ROI", value: "2,4x" },
];

export default function CinematicLanding() {
  const rootRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const root = rootRef.current;
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!root || prefersReducedMotion) return;

    let frame = 0;
    const updateScene = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const rect = root.getBoundingClientRect();
        const progress = Math.min(1, Math.max(0, Math.abs(rect.top) / Math.max(1, root.offsetHeight - window.innerHeight)));
        root.style.setProperty("--scroll-progress", progress.toFixed(3));
      });
    };

    updateScene();
    window.addEventListener("scroll", updateScene, { passive: true });
    window.addEventListener("resize", updateScene);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", updateScene);
      window.removeEventListener("resize", updateScene);
    };
  }, []);

  return (
    <main className="cinematic-landing" ref={rootRef}>
      <section className="cinematic-hero" aria-label="Landing de bienvenida">
        <div className="cinematic-bg" aria-hidden="true">
          <div className="cinematic-grid" />
          <div className="cinematic-orbit cinematic-orbit--one" />
          <div className="cinematic-orbit cinematic-orbit--two" />
          <div className="cinematic-core">
            <span />
            <span />
            <span />
          </div>
          <div className="cinematic-factory">
            <div className="factory-block factory-block--one" />
            <div className="factory-block factory-block--two" />
            <div className="factory-block factory-block--three" />
            <div className="factory-line factory-line--one" />
            <div className="factory-line factory-line--two" />
            <div className="factory-line factory-line--three" />
          </div>
        </div>

        <nav className="cinematic-nav">
          <div className="cinematic-brand">
            <span className="cinematic-brand__mark">D</span>
            <span>Decarb Decision Studio</span>
          </div>
          <div className="cinematic-nav__links">
            <a href="#cinematic-flow">Como funciona</a>
            <Link to="/tool">Herramienta</Link>
          </div>
        </nav>

        <div className="cinematic-hero__content">
          <p className="cinematic-kicker">Plataforma inteligente para industria baja en carbono</p>
          <h1>Herramienta de apoyo a la toma de decisiones para la descarbonizacion industrial</h1>
          <p>
            Calcula, analiza y optimiza iniciativas de reduccion de emisiones mediante una experiencia visual,
            inteligente e interactiva.
          </p>
          <div className="cinematic-actions">
            <Link className="cinematic-button cinematic-button--primary" to="/tool">
              Entrar en la herramienta
            </Link>
            <a className="cinematic-button cinematic-button--secondary" href="#cinematic-flow">
              Ver como funciona
            </a>
          </div>
        </div>

        <aside className="cinematic-console" aria-label="Indicadores principales">
          <div className="console-topline">
            <span>Decision engine</span>
            <i>Live model</i>
          </div>
          <div className="console-radar">
            <span />
            <span />
            <span />
          </div>
          <div className="console-signals">
            {signals.map((signal) => (
              <div key={signal.label}>
                <span>{signal.label}</span>
                <strong>{signal.value}</strong>
              </div>
            ))}
          </div>
        </aside>
      </section>

      <section className="cinematic-narrative" id="cinematic-flow">
        <div className="cinematic-section-copy">
          <p className="cinematic-kicker">Scroll cinematico</p>
          <h2>De datos operativos a decisiones de inversion climatica.</h2>
          <p>
            La escena conecta emisiones, contexto, iniciativas y restricciones para que las metricas funcionen como
            herramientas de decision, no como indicadores decorativos.
          </p>
        </div>

        <div className="floating-card-stage" aria-label="Capacidades principales">
          {floatingCards.map((card, index) => (
            <article className={`cinematic-floating-card cinematic-floating-card--${index + 1}`} key={card.title}>
              <span>{card.metric}</span>
              <h3>{card.title}</h3>
              <p>{card.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="cinematic-workflow" aria-label="Como funciona">
        <div className="cinematic-section-copy">
          <p className="cinematic-kicker">Metodo de trabajo</p>
          <h2>Como funciona</h2>
        </div>
        <ol className="cinematic-steps">
          {steps.map((step, index) => (
            <li key={step}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <p>{step}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="cinematic-final-cta">
        <p className="cinematic-kicker">Listo para decidir</p>
        <h2>Entra en la herramienta y construye un portafolio de descarbonizacion accionable.</h2>
        <Link className="cinematic-button cinematic-button--primary" to="/tool">
          Entrar en la herramienta
        </Link>
      </section>
    </main>
  );
}
