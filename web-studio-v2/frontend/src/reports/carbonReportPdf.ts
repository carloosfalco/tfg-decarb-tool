import type { AppState, Initiative } from "../types";

type PdfPage = { content: string[] };

const PAGE_WIDTH = 595;
const PAGE_HEIGHT = 842;
const MARGIN = 42;
const FOOTER_Y = 28;

const MOJIBAKE: Array<[RegExp, string]> = [
  [/Ã¡/g, "á"],
  [/Ã©/g, "é"],
  [/Ã­/g, "í"],
  [/Ã³/g, "ó"],
  [/Ãº/g, "ú"],
  [/Ã±/g, "ñ"],
  [/Ã/g, "Á"],
  [/Ã‰/g, "É"],
  [/Ã/g, "Í"],
  [/Ã“/g, "Ó"],
  [/Ãš/g, "Ú"],
  [/Ã‘/g, "Ñ"],
  [/â‚¬/g, "€"],
  [/Âº/g, "º"],
  [/â€“|â€”/g, "-"],
  [/â€¢/g, "•"],
  [/â‚‚/g, "2"],
];

function text(value: unknown) {
  let output = String(value ?? "");
  for (const [pattern, replacement] of MOJIBAKE) output = output.replace(pattern, replacement);
  return output.replace(/\*\*/g, "").replace(/\s+/g, " ").trim();
}

function slug(value: unknown) {
  return text(value)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x20-\x7E]/g, "");
}

function winAnsiByte(char: string) {
  const code = char.charCodeAt(0);
  if (code <= 0xff) return code;
  const replacements: Record<string, number> = { "€": 0x80, "•": 0x95, "–": 0x96, "—": 0x97 };
  return replacements[char] ?? "?".charCodeAt(0);
}

function pdfText(value: string) {
  const bytes = [...text(value)].map(winAnsiByte);
  return `<${bytes.map((byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase()}>`;
}

function formatNumber(value: unknown, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/D";
  return numeric.toLocaleString("es-ES", { maximumFractionDigits: digits });
}

function formatCurrency(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/D";
  return `${numeric.toLocaleString("es-ES", { maximumFractionDigits: 0 })} €`;
}

function formatPct(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/D";
  return `${(numeric * 100).toLocaleString("es-ES", { maximumFractionDigits: 1 })}%`;
}

function wrap(value: string, maxChars: number) {
  const words = text(value).split(" ").filter(Boolean);
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    if (!current) current = word;
    else if (`${current} ${word}`.length <= maxChars) current = `${current} ${word}`;
    else {
      lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  return lines.length ? lines : [""];
}

class PdfDocument {
  private pages: PdfPage[] = [];
  private page: PdfPage = this.newPage();
  private y = PAGE_HEIGHT - MARGIN;

  constructor(private readonly title: string) {}

  private newPage() {
    const page = { content: [] };
    this.pages.push(page);
    this.page = page;
    this.y = PAGE_HEIGHT - MARGIN;
    return page;
  }

  pageBreak() {
    if (this.y < PAGE_HEIGHT - MARGIN) this.newPage();
  }

  spacer(height: number) {
    this.ensure(height + 20);
    this.y -= height;
  }

  private ensure(height: number) {
    if (this.y - height < 70) this.newPage();
  }

  private setColor(stroke = "0 0 0", fill = "0 0 0") {
    this.page.content.push(`${stroke} RG ${fill} rg`);
  }

  private write(value: string, x: number, y: number, size: number, font = "F1", fill = "0.10 0.14 0.17") {
    this.setColor("0 0 0", fill);
    this.page.content.push(`BT /${font} ${size} Tf 1 0 0 1 ${x} ${y} Tm ${pdfText(value)} Tj ET`);
    this.setColor();
  }

  private rect(x: number, y: number, width: number, height: number, fill: string) {
    this.setColor("0 0 0", fill);
    this.page.content.push(`${x} ${y} ${width} ${height} re f`);
    this.setColor();
  }

  private line(x1: number, y1: number, x2: number, y2: number, width = 0.7, stroke = "0.82 0.86 0.88") {
    this.setColor(stroke);
    this.page.content.push(`${width} w ${x1} ${y1} m ${x2} ${y2} l S`);
    this.setColor();
  }

  cover(companyName: string, subtitle: string, meta: string[]) {
    this.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, "0.965 0.975 0.970");
    this.rect(0, 646, PAGE_WIDTH, 196, "0.050 0.360 0.330");
    this.rect(MARGIN, 558, PAGE_WIDTH - MARGIN * 2, 72, "1 1 1");
    this.write("INFORME DE DESCARBONIZACIÓN", MARGIN, 744, 18, "F2", "1 1 1");
    this.write(companyName || "Empresa sin nombre", MARGIN, 696, 31, "F2", "1 1 1");
    this.write(subtitle, MARGIN, 668, 12, "F1", "0.86 0.96 0.93");
    this.write("Documento ejecutivo", MARGIN + 18, 604, 15, "F2", "0.050 0.300 0.280");
    this.write("Huella, contexto estratégico e iniciativas priorizadas", MARGIN + 18, 582, 10, "F1", "0.25 0.31 0.35");
    meta.forEach((item, index) => this.write(item, MARGIN, 520 - index * 22, 11));
    this.line(MARGIN, 432, PAGE_WIDTH - MARGIN, 432, 1);
    this.newPage();
  }

  section(title: string) {
    this.ensure(54);
    this.write(title, MARGIN, this.y, 17, "F2", "0.050 0.300 0.280");
    this.line(MARGIN, this.y - 9, PAGE_WIDTH - MARGIN, this.y - 9, 1);
    this.y -= 32;
  }

  subheading(title: string) {
    this.ensure(28);
    this.write(title, MARGIN, this.y, 12, "F2");
    this.y -= 18;
  }

  paragraph(value: string) {
    const lines = wrap(value, 94);
    this.ensure(lines.length * 13 + 8);
    for (const line of lines) {
      this.write(line, MARGIN, this.y, 9.5);
      this.y -= 13;
    }
    this.y -= 5;
  }

  bullets(items: string[]) {
    for (const item of items.filter(Boolean)) {
      const lines = wrap(item, 90);
      this.ensure(lines.length * 12 + 6);
      lines.forEach((line, index) => {
        this.write(index === 0 ? `• ${line}` : `  ${line}`, MARGIN, this.y, 9);
        this.y -= 12;
      });
      this.y -= 4;
    }
  }

  keyValues(rows: Array<[string, string]>) {
    rows.forEach(([label, value], index) => {
      this.ensure(23);
      if (index % 2 === 0) this.rect(MARGIN - 7, this.y - 8, PAGE_WIDTH - MARGIN * 2 + 14, 21, "0.985 0.990 0.990");
      this.write(label, MARGIN, this.y, 9, "F2", "0.25 0.31 0.35");
      this.write(value, 245, this.y, 9);
      this.y -= 21;
    });
    this.y -= 8;
  }

  metricCards(cards: Array<[string, string]>) {
    const cardWidth = (PAGE_WIDTH - MARGIN * 2 - 18) / 3;
    this.ensure(76);
    cards.slice(0, 3).forEach(([label, value], index) => {
      const x = MARGIN + index * (cardWidth + 9);
      this.rect(x, this.y - 56, cardWidth, 58, "0.940 0.975 0.965");
      this.write(value, x + 10, this.y - 22, 14, "F2", "0.050 0.360 0.330");
      this.write(label, x + 10, this.y - 42, 7.5, "F2", "0.35 0.40 0.43");
    });
    this.y -= 78;
  }

  table(headers: string[], rows: string[][], widths: number[], options?: { fontSize?: number; firstColChars?: number; otherChars?: number }) {
    const fontSize = options?.fontSize ?? 8;
    this.ensure(40);
    this.rect(MARGIN, this.y - 18, PAGE_WIDTH - MARGIN * 2, 24, "0.050 0.360 0.330");
    let x = MARGIN + 5;
    headers.forEach((header, index) => {
      this.write(header, x, this.y - 9, fontSize, "F2", "1 1 1");
      x += widths[index];
    });
    this.y -= 30;

    rows.forEach((row, rowIndex) => {
      this.ensure(24);
      if (rowIndex % 2 === 0) this.rect(MARGIN, this.y - 8, PAGE_WIDTH - MARGIN * 2, 20, "0.985 0.990 0.990");
      x = MARGIN + 5;
      row.forEach((cell, index) => {
        const maxChars = index === 0 ? options?.firstColChars ?? 46 : options?.otherChars ?? 18;
        this.write(text(cell).slice(0, maxChars), x, this.y, fontSize);
        x += widths[index];
      });
      this.y -= 20;
    });
    this.y -= 10;
  }

  initiativeTable(rows: Initiative[]) {
    this.ensure(36);
    this.rect(MARGIN, this.y - 18, PAGE_WIDTH - MARGIN * 2, 24, "0.050 0.360 0.330");
    this.write("Iniciativas y métricas principales", MARGIN + 7, this.y - 9, 8, "F2", "1 1 1");
    this.y -= 32;

    rows.forEach((item, index) => {
      this.ensure(76);
      if (index % 2 === 0) this.rect(MARGIN, this.y - 56, PAGE_WIDTH - MARGIN * 2, 64, "0.985 0.990 0.990");
      this.write(`${index + 1}. ${item.initiative}`, MARGIN + 7, this.y, 8.4, "F2", "0.10 0.14 0.17");
      this.y -= 13;
      this.write(`Scope: ${item.scope || "N/D"}   |   Familia: ${item.initiative_family || "N/D"}   |   Fuente: ${item.emission_source || "N/D"}`, MARGIN + 7, this.y, 7.2, "F1", "0.35 0.40 0.43");
      this.y -= 15;
      this.write(`CAPEX: ${formatCurrency(item.capex_eur)}`, MARGIN + 7, this.y, 7.4, "F2");
      this.write(`CO2: ${formatNumber(item.annual_co2_reduction_t)} t/año`, MARGIN + 104, this.y, 7.4, "F2");
      this.write(`OPEX: ${formatCurrency(item.annual_opex_saving_eur)}`, MARGIN + 192, this.y, 7.4, "F2");
      this.write(`Meses: ${formatNumber(item.implementation_months, 0)}`, MARGIN + 296, this.y, 7.4, "F2");
      this.write(
        `Payback: ${Number.isFinite(Number(item.payback_years)) ? `${formatNumber(item.payback_years)} años` : "N/D"}`,
        MARGIN + 360,
        this.y,
        7.4,
        "F2"
      );
      this.write(`VAN: ${formatCurrency(item.npv_eur)}`, MARGIN + 448, this.y, 7.4, "F2");
      this.y -= 18;
    });
  }

  build() {
    const objects: string[] = [];
    const pageIds: number[] = [];
    const contentIds: number[] = [];
    const catalogId = 1;
    const pagesId = 2;
    let nextId = 3;
    for (let i = 0; i < this.pages.length; i += 1) {
      pageIds.push(nextId++);
      contentIds.push(nextId++);
    }
    const regularFontId = nextId++;
    const boldFontId = nextId++;

    objects[catalogId] = `<< /Type /Catalog /Pages ${pagesId} 0 R >>`;
    objects[pagesId] = `<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(" ")}] /Count ${pageIds.length} >>`;

    this.pages.forEach((page, index) => {
      page.content.push(`0.82 0.86 0.88 RG 0.4 w ${MARGIN} 44 m ${PAGE_WIDTH - MARGIN} 44 l S`);
      page.content.push(`BT /F1 8 Tf 1 0 0 1 ${MARGIN} ${FOOTER_Y} Tm ${pdfText(this.title)} Tj ET`);
      page.content.push(`BT /F1 8 Tf 1 0 0 1 ${PAGE_WIDTH - 92} ${FOOTER_Y} Tm ${pdfText(`${index + 1} / ${this.pages.length}`)} Tj ET`);
      const stream = page.content.join("\n");
      objects[pageIds[index]] =
        `<< /Type /Page /Parent ${pagesId} 0 R /MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}] /Resources << /Font << /F1 ${regularFontId} 0 R /F2 ${boldFontId} 0 R >> >> /Contents ${contentIds[index]} 0 R >>`;
      objects[contentIds[index]] = `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`;
    });

    objects[regularFontId] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";
    objects[boldFontId] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>";

    let pdf = "%PDF-1.4\n";
    const offsets = [0];
    for (let id = 1; id < objects.length; id += 1) {
      offsets[id] = pdf.length;
      pdf += `${id} 0 obj\n${objects[id]}\nendobj\n`;
    }
    const xrefOffset = pdf.length;
    pdf += `xref\n0 ${objects.length}\n0000000000 65535 f \n`;
    for (let id = 1; id < objects.length; id += 1) pdf += `${String(offsets[id]).padStart(10, "0")} 00000 n \n`;
    pdf += `trailer\n<< /Size ${objects.length} /Root ${catalogId} 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`;
    return new Blob([pdf], { type: "application/pdf" });
  }
}

function selectedInitiatives(state: AppState) {
  const selected = (state.portfolioResult?.initiatives ?? []).filter((item) => item.selected);
  if (selected.length) return selected;
  return state.metricsResult.length ? state.metricsResult.slice(0, 8) : state.initiatives.slice(0, 8);
}

function mainEmissionSource(state: AppState) {
  const entries = Object.entries(state.footprint?.breakdown ?? {});
  if (!entries.length) return "No disponible";
  const [name, value] = entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0];
  return `${name}: ${formatNumber(value)} tCO2e`;
}

export function buildCarbonReportPdf(state: AppState) {
  const company = state.companyInputs;
  const docTitle = `Informe de descarbonización - ${company.company_name || "empresa"}`;
  const doc = new PdfDocument(docTitle);
  const selected = selectedInitiatives(state);
  const selectedCo2 = selected.reduce((sum, item) => sum + Number(item.annual_co2_reduction_t ?? 0), 0);
  const selectedCapex = selected.reduce((sum, item) => sum + Number(item.capex_eur ?? 0), 0);

  doc.cover(company.company_name, "Huella de carbono, PESTEL y portafolio de iniciativas", [
    `Ubicación: ${company.province || "N/D"} - ${company.country || company.country_region || "N/D"}`,
    `Año de inventario: ${company.inventory_year}`,
    `Fecha de generación: ${new Date().toLocaleString("es-ES")}`,
  ]);

  doc.section("Resumen ejecutivo");
  doc.metricCards([
    ["Huella total", state.footprint ? `${formatNumber(state.footprint.total_t)} tCO2e/año` : "Pendiente"],
    ["CAPEX seleccionado", state.portfolioResult ? formatCurrency(state.portfolioResult.summary.total_capex) : formatCurrency(selectedCapex)],
    ["Reducción anual", state.portfolioResult ? `${formatNumber(state.portfolioResult.summary.total_co2)} tCO2e` : `${formatNumber(selectedCo2)} tCO2e`],
  ]);
  doc.keyValues([
    ["Empresa", company.company_name || "N/D"],
    ["Sector", company.sector || "N/D"],
    ["Alcance 1", state.footprint ? `${formatNumber(state.footprint.scope1_t)} tCO2e/año` : "N/D"],
    ["Alcance 2", state.footprint ? `${formatNumber(state.footprint.scope2_t)} tCO2e/año` : "N/D"],
    ["Principal fuente", mainEmissionSource(state)],
    ["Iniciativas en cartera", String(state.initiatives.length)],
    ["Iniciativas seleccionadas", String(state.portfolioResult?.summary.selected_count ?? selected.filter((item) => item.selected).length)],
  ]);

  doc.section("Huella de carbono");
  if (state.footprint) {
    doc.keyValues([
      ["Total", `${formatNumber(state.footprint.total_t)} tCO2e/año`],
      ["Alcance 1", `${formatNumber(state.footprint.scope1_t)} tCO2e/año`],
      ["Alcance 2", `${formatNumber(state.footprint.scope2_t)} tCO2e/año`],
      ["Método electricidad", state.footprint.scope2_elec_method || "N/D"],
      ["Calidad del dato", `${state.footprint.quality.label}: ${state.footprint.quality.text}`],
    ]);
    doc.table(
      ["Fuente", "tCO2e"],
      Object.entries(state.footprint.breakdown ?? {}).map(([label, value]) => [label, formatNumber(value)]),
      [365, 120]
    );
    doc.subheading("Trazabilidad de factores");
    doc.bullets([
      `Alcance 1: ${state.footprint.scope1_factor_source ?? "N/D"}`,
      `Electricidad: ${state.footprint.scope2_elec_source ?? "N/D"}`,
      `Calor / vapor: ${state.footprint.scope2_heat_source ?? "N/D"}`,
      ...(state.footprint.scope2_notes ?? []),
      ...(state.footprint.scope2_errors ?? []),
    ]);
  } else {
    doc.paragraph("La huella todavía no está calculada en la aplicación. Esta sección queda pendiente.");
  }

  doc.pageBreak();
  doc.section("Análisis PESTEL");
  const pestelEntries = Object.entries(state.pestel ?? {});
  if (pestelEntries.length) {
    for (const [category, items] of pestelEntries) {
      doc.subheading(category);
      doc.bullets((items ?? []).slice(0, 5));
    }
  } else {
    doc.paragraph("No hay PESTEL disponible en el estado actual. Genera el análisis una vez en la app y vuelve a exportar el PDF.");
  }

  doc.pageBreak();
  doc.section("Portafolio de iniciativas");
  if (state.portfolioResult) {
    doc.keyValues([
      ["Estado optimización", state.portfolioResult.summary.status],
      ["Seleccionadas", String(state.portfolioResult.summary.selected_count)],
      ["CAPEX total", formatCurrency(state.portfolioResult.summary.total_capex)],
      ["Reducción anual", `${formatNumber(state.portfolioResult.summary.total_co2)} tCO2e`],
      ["VAN total", formatCurrency(state.portfolioResult.summary.total_npv)],
    ]);
  } else {
    doc.paragraph("El portafolio aún no está optimizado. Se muestran las iniciativas disponibles o las métricas calculadas si existen.");
  }
  if (selected.length) doc.initiativeTable(selected);
  else doc.paragraph("No hay iniciativas disponibles en el estado actual.");

  doc.spacer(18);
  doc.section("Supuestos financieros y trazabilidad IA");
  doc.keyValues([
    ["Tasa de descuento", formatPct(state.financialParams.discount_rate)],
    ["Horizonte", `${state.financialParams.horizon} años`],
    ["Payback máximo", state.financialParams.max_payback_years ? `${state.financialParams.max_payback_years} años` : "Sin límite"],
    ["Modelo PESTEL", state.pestelMeta?.model ?? "N/D"],
    ["Fecha PESTEL", state.pestelMeta?.generated_at ?? "N/D"],
    ["Modelo iniciativas", state.initiativesMeta?.model ?? "N/D"],
    ["Fecha iniciativas", state.initiativesMeta?.generated_at ?? "N/D"],
  ]);

  return doc.build();
}

export function downloadCarbonReportPdf(state: AppState) {
  const blob = buildCarbonReportPdf(state);
  const companySlug = slug(state.companyInputs.company_name || "empresa")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `informe-descarbonizacion-${companySlug || "empresa"}.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
