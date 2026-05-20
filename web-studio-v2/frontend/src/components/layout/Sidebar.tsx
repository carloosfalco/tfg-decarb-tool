import type { StepId, StepStatus } from "../../types";

const steps: { id: StepId; label: string }[] = [
  { id: "empresa", label: "Empresa" },
  { id: "alcance1", label: "Alcance 1" },
  { id: "alcance2", label: "Alcance 2" },
  { id: "pestel", label: "PESTEL" },
  { id: "iniciativas", label: "Iniciativas" },
  { id: "portafolio", label: "Portafolio" },
];

export function Sidebar({
  currentStep,
  statuses,
  onStepChange,
}: {
  currentStep: StepId;
  statuses: Record<StepId, StepStatus>;
  onStepChange: (step: StepId) => void;
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__eyebrow">Industrial Decarbonization Tool</span>
        <h1>Descarbonización industrial</h1>
        <p>Flujo técnico para cálculo, contexto estratégico, evaluación financiera y selección de portfolio.</p>
      </div>
      <nav className="sidebar__nav">
        {steps.map((step, index) => {
          const status = statuses[step.id];
          return (
            <button
              key={step.id}
              type="button"
              className={`sidebar__step ${currentStep === step.id ? "is-active" : ""} is-${status}`}
              onClick={() => onStepChange(step.id)}
            >
              <span className="sidebar__step-index">{index + 1}</span>
              <span>
                <strong>{step.label}</strong>
                <small>{status.replace("-", " ")}</small>
              </span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
