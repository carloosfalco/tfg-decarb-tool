import { useEffect } from "react";
import { useForm } from "react-hook-form";
import type { CompanyInputs, MeasureStatus } from "../../types";

const measures = [
  "LED",
  "GdO",
  "Paneles solares",
  "Flota eléctrica",
  "Variadores de frecuencia",
  "EMS/submetering",
  "Recuperación de calor",
  "Programa de fugas de aire comprimido",
] as const;

type CompanyFormValues = Pick<
  CompanyInputs,
  | "company_name"
  | "sector"
  | "country"
  | "province"
  | "postal_code"
  | "main_customer_locations"
  | "inventory_year"
  | "electricity_price_eur_mwh"
  | "roof_area_m2"
> & {
  implemented_measures: Record<string, MeasureStatus>;
};

export function CompanyForm({
  value,
  onChange,
  onSubmit,
}: {
  value: CompanyInputs;
  onChange?: (value: CompanyInputs) => void;
  onSubmit: (value: CompanyInputs) => void;
}) {
  const { register, handleSubmit, watch, setValue } = useForm<CompanyFormValues>({
    defaultValues: {
      company_name: value.company_name,
      sector: value.sector,
      country: value.country ?? "España",
      province: value.province,
      postal_code: value.postal_code ?? "",
      main_customer_locations: value.main_customer_locations ?? "",
      inventory_year: value.inventory_year,
      electricity_price_eur_mwh: value.electricity_price_eur_mwh || undefined,
      roof_area_m2: value.roof_area_m2,
      implemented_measures: value.implemented_measures,
    },
  });

  const measuresState = watch("implemented_measures");
  const province = watch("province");

  function buildCompanyInputs(formValues: CompanyFormValues): CompanyInputs {
    return {
      ...value,
      ...formValues,
      country: "España",
      country_region: `España - ${formValues.province || province || "Provincia"}`,
      electricity_price_eur_mwh: Number.isFinite(formValues.electricity_price_eur_mwh)
        ? formValues.electricity_price_eur_mwh
        : 0,
      has_invoices: false,
      has_meters: false,
      has_submetering: false,
      has_energy_audit: false,
    };
  }

  useEffect(() => {
    const subscription = watch((formValues) => {
      onChange?.(buildCompanyInputs(formValues as CompanyFormValues));
    });
    return () => subscription.unsubscribe();
  }, [onChange, watch, value, province]);

  return (
    <form
      className="panel form-grid company-form"
      onSubmit={handleSubmit((formValues) => onSubmit(buildCompanyInputs(formValues)))}
    >
      <div className="form-grid__three">
        <label>
          <span>Nombre de la organización</span>
          <input {...register("company_name")} placeholder="Nombre legal o comercial" />
        </label>
        <label>
          <span>CNAE / sector</span>
          <input {...register("sector")} placeholder="Actividad principal, proceso o sector" />
        </label>
        <label>
          <span>Año de inventario</span>
          <select {...register("inventory_year", { valueAsNumber: true })}>
            <option value={2023}>2023</option>
            <option value={2024}>2024</option>
            <option value={2025}>2025</option>
          </select>
        </label>
      </div>

      <div className="form-grid__three">
        <label>
          <span>País</span>
          <input value="España" disabled />
        </label>
        <label>
          <span>Provincia</span>
          <input {...register("province")} placeholder="Provincia de operación principal" />
        </label>
        <label>
          <span>Código postal</span>
          <input {...register("postal_code")} placeholder="Código postal" maxLength={5} />
        </label>
      </div>

      <label>
        <span>Ubicación de principales clientes o mercados</span>
        <input
          {...register("main_customer_locations")}
          placeholder="Ej. Comunidad Valenciana, España, UE, norte de África..."
        />
      </label>

      <div className="form-grid__two">
        <label>
          <span>Precio electricidad opcional</span>
          <div className="input-with-unit">
            <input
              type="number"
              step="1"
              placeholder="Se estimará si se deja vacío"
              {...register("electricity_price_eur_mwh", { valueAsNumber: true })}
            />
            <small>€/MWh</small>
          </div>
        </label>
        <label>
          <span>Área disponible de cubierta</span>
          <div className="input-with-unit">
            <input type="number" step="1" {...register("roof_area_m2", { valueAsNumber: true })} />
            <small>m²</small>
          </div>
        </label>
      </div>

      <div>
        <h3>Medidas ya implantadas</h3>
        <div className="checkbox-grid">
          {measures.map((label) => (
            <label key={label} className="check-row">
              <input
                type="checkbox"
                checked={(measuresState?.[label] ?? "No") === "Sí"}
                onChange={(event) =>
                  setValue(`implemented_measures.${label}` as const, event.target.checked ? "Sí" : "No")
                }
              />
              {label}
            </label>
          ))}
        </div>
      </div>

      <div className="form-actions">
        <button className="button button--primary" type="submit">
          Siguiente: Alcance 1
        </button>
      </div>
    </form>
  );
}

