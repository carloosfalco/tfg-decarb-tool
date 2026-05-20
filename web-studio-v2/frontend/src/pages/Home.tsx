import {
  Target,
  Factory,
  Plug,
  Network,
  BrainCircuit,
  BarChart3,
  Calculator,
  Globe2,
  Leaf,
  CircleDollarSign,
  ListChecks,
  Download,
  Building2,
  Gauge,
  CheckCircle2,
  Rocket,
} from "lucide-react";
import { Link } from "react-router-dom";

type LandingIcon = typeof Target;

function IconBubble({
  icon: Icon,
  tone,
  large = false,
}: {
  icon: LandingIcon;
  tone: string;
  large?: boolean;
}) {
  return (
    <span className={`icon-bubble ${large ? "icon-bubble--large " : ""}${tone}`}>
      <Icon className="landing-icon" aria-hidden="true" />
    </span>
  );
}

const heroItems: Array<[LandingIcon, string, string, string]> = [
  [Target, "tone-green", "Enfoque en Alcance 1 y 2", "Cálculo de emisiones directas e indirectas por energía."],
  [BrainCircuit, "tone-blue", "Inteligencia artificial", "Análisis PESTEL e iniciativas generadas con IA."],
  [BarChart3, "tone-orange", "Decisiones basadas en datos", "Evaluación y optimización del portafolio de medidas."],
];

const outputItems: Array<[LandingIcon, string, string, string]> = [
  [Calculator, "tone-green", "Cálculo de huella", "Emisiones de Alcance 1 y 2 con desglose por fuente y categoría."],
  [Globe2, "tone-blue", "Análisis PESTEL", "Contexto político, económico, social, tecnológico, ambiental y legal."],
  [Leaf, "tone-teal", "Propuestas de reducción", "Iniciativas de descarbonización adaptadas al perfil de la empresa."],
  [CircleDollarSign, "tone-orange", "Evaluación financiera", "Coste, ahorro esperado, retorno, VAN y eficiencia por tonelada evitada."],
  [ListChecks, "tone-purple", "Priorización de medidas", "Selección de iniciativas según presupuesto, impacto y viabilidad."],
  [Download, "tone-amber", "Resultados descargables", "Tablas exportables para documentar el análisis y defender las decisiones."],
];

const workflowSteps: Array<[string, LandingIcon, string, string]> = [
  ["01", Building2, "Datos de empresa", "Sector, ubicación, año de inventario y datos de actividad."],
  ["02", Calculator, "Cálculo de emisiones", "Consumos, combustibles, electricidad, vehículos y refrigerantes."],
  ["03", Gauge, "Diagnóstico", "Resultados de huella y comparación entre métodos de cálculo."],
  ["04", Leaf, "Propuestas", "Medidas de reducción apoyadas por inteligencia artificial."],
  ["05", CheckCircle2, "Decisión final", "Evaluación económica y selección del portafolio recomendado."],
];

export default function Home() {
  return (
    <main className="landing landing--product">
      <nav className="landing-nav">
        <div className="landing-brand">
          <strong>TFG Carlos Falcó Caravajal</strong>
        </div>
        <div className="landing-nav__links">
          <a href="#conceptos">Antes de empezar</a>
          <a href="#funcionamiento">Cómo funciona</a>
        </div>
      </nav>

      <section className="product-hero">
        <div className="product-hero__copy">
          <h1>Herramienta de cálculo de huella de carbono y propuestas de descarbonización empresarial</h1>
          <p>Calcula emisiones, analiza el contexto de la empresa y prioriza iniciativas de reducción.</p>
          <Link className="product-cta button" to="/tool">
            Entrar en la herramienta →
          </Link>
        </div>

        <aside className="hero-summary" aria-label="Capacidades principales">
          {heroItems.map(([icon, tone, title, text]) => (
            <div className="hero-summary__item" key={title}>
              <IconBubble icon={icon} tone={tone} />
              <div>
                <strong>{title}</strong>
                <p>{text}</p>
              </div>
            </div>
          ))}
        </aside>
      </section>

      <section className="landing-product-section" id="conceptos">
        <div className="section-title">
          <div>
            <h2>Conceptos clave antes de empezar</h2>
            <p>La huella de carbono de una organización se clasifica habitualmente en tres alcances de emisiones.</p>
          </div>
        </div>

        <div className="scope-product-grid">
          <article className="scope-product-card">
            <IconBubble icon={Factory} tone="tone-green" large />
            <h3>Alcance 1</h3>
            <p>Emisiones directas generadas por fuentes propiedad de la empresa o bajo su control.</p>
            <ul>
              <li>Combustibles</li>
              <li>Vehículos propios</li>
              <li>Fugas de refrigerantes</li>
            </ul>
          </article>
          <article className="scope-product-card">
            <IconBubble icon={Plug} tone="tone-blue" large />
            <h3>Alcance 2</h3>
            <p>Emisiones indirectas asociadas al consumo de energía adquirida por la empresa.</p>
            <ul>
              <li>Electricidad comprada</li>
              <li>Energía adquirida</li>
              <li>Comercializadoras eléctricas</li>
            </ul>
          </article>
          <article className="scope-product-card">
            <IconBubble icon={Network} tone="tone-purple" large />
            <h3>Alcance 3</h3>
            <p>Otras emisiones indirectas que se producen a lo largo de la cadena de valor.</p>
            <ul>
              <li>Compras a proveedores</li>
              <li>Transporte externo y viajes</li>
              <li>Residuos o uso de productos vendidos</li>
            </ul>
          </article>
        </div>
      </section>

      <section className="landing-product-section" id="resultados">
        <div className="section-title">
          <div>
            <h2>Qué obtendrás con la herramienta</h2>
            <p>La herramienta convierte los datos introducidos en resultados técnicos, estratégicos y económicos.</p>
          </div>
        </div>

        <div className="output-product-grid">
          {outputItems.map(([icon, tone, title, text]) => (
            <article className="output-product-card" key={title}>
              <IconBubble icon={icon} tone={tone} />
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-product-section workflow-product-section" id="funcionamiento">
        <div className="section-title">
          <div>
            <h2>Cómo funciona</h2>
            <p>La aplicación sigue una secuencia lógica desde la recogida de datos hasta la propuesta final.</p>
          </div>
        </div>

        <ol className="workflow-product-steps">
          {workflowSteps.map(([number, icon, title, text]) => {
            const StepIcon = icon;
            return (
              <li key={number}>
                <span className="workflow-number">{number}</span>
                <div className="workflow-step-card">
                  <StepIcon className="landing-icon" aria-hidden="true" />
                  <h3>{title}</h3>
                  <p>{text}</p>
                </div>
              </li>
            );
          })}
        </ol>
      </section>

      <aside className="mvp-product-note">
        <IconBubble icon={Rocket} tone="tone-green" large />
        <div>
          <strong>Prototipo funcional para toma de decisiones empresariales</strong>
          <p>
            Esta herramienta está planteada como un MVP académico aplicado: combina cálculo ambiental, análisis estratégico
            y priorización económica para ayudar a una empresa a construir una hoja de ruta inicial de descarbonización.
          </p>
        </div>
      </aside>
    </main>
  );
}
