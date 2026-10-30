"use strict";
// Adapted from PersEval final: preserve the UI; the site build supplies every rank.

const METRIC_FIELDS = ["global_f1", "user_f1", "text_f1"];
const NUMERIC_FIELDS = new Set(["rank", ...METRIC_FIELDS]);
const DATASET_ORDER = ["BREXIT", "DICES", "EPIC", "MD", "MHS"];
const VARIANT_ORDER = [
  "baseline::none",
  "named::none",
  "named::train",
  "unnamed::train",
  "named::test",
  "unnamed::test",
];
const TASK_LABELS = {
  baseline: "Baseline",
  named: "Named perspective",
  unnamed: "Unnamed perspective",
};
const ADAPTATION_LABELS = {
  none: "No adaptation",
  train: "Training-time adaptation",
  test: "Inference-time adaptation",
};

const fieldValue = (entry, field) => {
  if (field === "family") {
    const family = entry.experiment?.model_family || entry.configuration?.hyperparameters?.inference?.model_family;
    return typeof family === "string" && family.trim() ? family : "Unspecified";
  }
  if (field === "source") return entry.source_type;
  if (field === "model") return entry.model;
  if (field === "scope") return entry.experiment?.scope || "main";
  if (field === "group") return entry.ranking_group_id;
  return String(entry.comparison_group[field]);
};
function selectEntries(entries, filters) {
  const query = (filters.search || "").trim().toLowerCase();
  return entries.filter((entry) => ["dataset", "label", "family", "task", "adaptation", "extended", "source", "group", "model", "scope"]
    .every((field) => !filters[field] || fieldValue(entry, field) === filters[field])
    && [entry.name, entry.model, entry.submitter, entry.submission_id, entry.experiment?.method, entry.configuration?.method, entry.experiment?.representation, entry.configuration?.representation].join(" ").toLowerCase().includes(query));
}
function sortEntries(entries, sort) {
  return [...entries].sort((a, b) => {
    if (sort === "name") return a.name.localeCompare(b.name) || a.submission_id.localeCompare(b.submission_id);
    if (sort === "rank") return a.rank - b.rank || a.submission_id.localeCompare(b.submission_id);
    const av = a.metrics[sort]?.mean, bv = b.metrics[sort]?.mean;
    if (av === null || av === undefined) return bv === null || bv === undefined ? 0 : 1;
    if (bv === null || bv === undefined) return -1;
    return bv - av || a.submission_id.localeCompare(b.submission_id);
  });
}
function metricText(metric, source, digits = 4) {
  if (metric.mean === null) return "Unavailable";
  const value = metric.mean.toFixed(digits);
  if (source === "reported") return value;
  return metric.stddev === null ? `${value} · SD unavailable` : `${value} ± ${metric.stddev.toFixed(digits)}`;
}
function groupLabel(group) {
  return [group.dataset, group.label, group.task, group.adaptation, `extended=${group.extended}`, group.split_protocol, group.scoring_rule].join(" · ");
}
function allowedHref(href) {
  // Render source strings as text; only known safe protocols/relative paths become links.
  if (typeof href !== "string" || /[\\\s\x00-\x1f]/u.test(href)) return false;
  if (/^https?:\/\//u.test(href)) {
    try { const url = new URL(href); return Boolean(url.hostname) && !url.username && !url.password; } catch { return false; }
  }
  return /^(?:[A-Za-z0-9][A-Za-z0-9_.-]*\/)*[A-Za-z0-9][A-Za-z0-9_.-]*$/u.test(href);
}

function familyName(value) {
  const normalized = String(value).toLowerCase();
  return normalized === "encoder" ? "Encoder" : normalized === "decoder" ? "Decoder" : "Unspecified";
}
function methodName(value, entry) {
  if (entry.source_type === "reported" && entry.comparison_group.task === "baseline" && familyName(fieldValue(entry, "family")) === "Encoder") return "Text-only baseline";
  const labels = {"focal-loss":"Fine-tuned encoder","base-zero":"Base-zero","perspective":"Perspective prompting","ipa":"IPA (LaMP)","ipa-lamp":"IPA (LaMP)","perspectivist-encoder":"Fine-tuned encoder"};
  return Object.hasOwn(labels, value) ? labels[value] : value || "Unspecified";
}
function representationName(entry) {
  const value = entry.experiment?.representation || entry.configuration?.representation;
  if (entry.experiment?.scope === "ablation" && entry.experiment.trait_subset?.length) return entry.experiment.trait_subset.join(" + ");
  const labels = {"user-id":"User ID","traits":"Traits","user-id+traits":"User ID + traits","user-history-and-traits":"Traits + user history","user-history":"User history","text-only":"Text only","none":"—"};
  return Object.hasOwn(labels, value) ? labels[value] : value || "—";
}
function compareGroups(left, right) {
  const datasetPosition = value => { const index = DATASET_ORDER.indexOf(value); return index < 0 ? DATASET_ORDER.length : index; };
  const variantPosition = group => { const index = VARIANT_ORDER.indexOf(group.task + "::" + group.adaptation); return index < 0 ? VARIANT_ORDER.length : index; };
  return datasetPosition(left.dataset) - datasetPosition(right.dataset)
    || left.dataset.localeCompare(right.dataset) || left.label.localeCompare(right.label)
    || Number(left.extended === "true") - Number(right.extended === "true")
    || variantPosition(left) - variantPosition(right) || left.key.localeCompare(right.key);
}
function enrichRows(entries) {
  const rows = entries.map((entry, sourceIndex) => ({
    ...entry.comparison_group,
    extended: String(entry.comparison_group.extended),
    model: entry.model,
    model_family: familyName(fieldValue(entry, "family")),
    method: methodName(entry.experiment?.method || entry.configuration?.method, entry),
    representation: representationName(entry),
    global_f1: entry.metrics.global_f1.mean,
    user_f1: entry.metrics.user_f1.mean,
    text_f1: entry.metrics.text_f1.mean,
    rank: entry.rank,
    groupKey: entry.ranking_group_id,
    sourceIndex,
    entry,
  }));
  const map = new Map();
  for (const row of rows) {
    if (!map.has(row.groupKey)) map.set(row.groupKey, {
      ...row.entry.comparison_group, extended: row.extended, key: row.groupKey,
      comparability: row.entry.comparability, rows: [], ranges: {},
    });
    map.get(row.groupKey).rows.push(row);
  }
  const groups = [...map.values()].sort(compareGroups);
  for (const group of groups) {
    for (const field of METRIC_FIELDS) {
      const values = group.rows.map(row => row[field]).filter(value => Number.isFinite(value));
      group.ranges[field] = {min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null};
    }
  }
  return {rows, groups};
}
function relativeHeatStyle(value, range, singleConfiguration = false) {
  if (!Number.isFinite(value) || range.min === null) return {hue:77, equal:true};
  if (singleConfiguration) return {hue:142, equal:false};
  if (range.max === range.min) return {hue:77, equal:true};
  return {hue:8 + (value - range.min) / (range.max - range.min) * 134, equal:false};
}
function csvCell(value) {
  let text = value === null || value === undefined ? "" : String(value);
  if (/^[\s]*[=+@-]|^[\t\r]/u.test(text)) text = "'" + text;
  return /[",\n\r]/u.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
}
function filteredCsv(rows) {
  const headers = ["submission_id","name","submitter","source_type","ranking_group_id","rank",
    "dataset","label","model_family","model","task","adaptation","extended","method","representation",
    "global_f1","global_f1_stddev","user_f1","user_f1_stddev","text_f1","text_f1_stddev","n","seeds"];
  return [headers.join(","), ...rows.map(row => {
    const entry = row.entry;
    const values = {...row, ...Object.fromEntries(["submission_id","name","submitter","source_type","ranking_group_id"].map(key => [key, entry[key]])),
      n: entry.source_type === "measured" ? entry.run_count : null,
      seeds: entry.source_type === "measured" ? entry.seeds.map(seed => seed ?? "unavailable").join(";") : "",
      ...Object.fromEntries(METRIC_FIELDS.map(field => [field + "_stddev", entry.metrics[field].stddev]))};
    return headers.map(header => csvCell(values[header])).join(",");
  })].join("\n") + "\n";
}

if (typeof module !== "undefined") module.exports = {selectEntries,sortEntries,metricText,fieldValue,allowedHref,enrichRows,relativeHeatStyle,filteredCsv};

if (typeof document !== "undefined") {
const byId = id => document.getElementById(id);
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function link(text, href, download = false) {
    if (!allowedHref(href)) return node("span", text);
    const el = node("a", text); el.href = href;
    if (download) el.download = typeof download === "string" ? download : "";
    if (/^https?:/u.test(href)) el.rel = "noopener noreferrer";
    return el;
  }
  function disclosure(title, value) {
    const details = node("details"); details.append(node("summary", title), node("pre", JSON.stringify(value, null, 2))); return details;
  }
  function runDetails(rid, data) {
    const run = data.runs[rid];
    const details = node("details", undefined, "run-detail");
    details.append(node("summary", `Run ${rid} · execution seed ${run.run.seed ?? "unavailable"}`));
    details.append(node("p", `Started ${run.run.execution.started_at}; finished ${run.run.execution.finished_at}. Split seed ${run.run.configuration.split.seed ?? "unavailable"}.`));
    details.append(disclosure("Configuration and provenance", run.run), disclosure("Compact metrics", run.metrics), disclosure("Environment snapshot", run.environment));
    const artifacts = data.artifact_metadata[rid] || [];
    if (!artifacts.length) details.append(node("p", "No optional artifacts were selected for this run."));
    for (const item of artifacts) {
      const block = node("p", `${item.description} · ${item.path} · ${item.size === null ? "size unavailable" : `${item.size.toLocaleString()} bytes`} · ${item.sha256}`);
      if (item.public) {
        block.append(node("br"), link(`Download artifact (${item.license})`, item.href, item.path.split("/").pop()), node("span", ` · ${item.attribution}`));
        if (item.path === "artifacts/perseval-evaluation.json") {
          const button = node("button", "Show complete and trait metrics"); button.type = "button";
          button.addEventListener("click", async () => {
            button.disabled = true;
            try {
              block.append(disclosure("Complete global, user, text and available trait metrics", run.complete_metrics));
              button.remove();
            } catch (error) { button.disabled = false; block.append(node("span", ` Unable to load metrics: ${error.message}`, "limitation")); }
          }); block.append(node("br"), button);
        }
      } else block.append(node("span", " · Private; not served by this website."));
      details.append(block);
    }
    return details;
  }
  function entryDetails(entry, data) {
    const details = node("details", undefined, "entry-detail"); details.append(node("summary", "Submission details"));
    details.append(node("h3", entry.name), node("p", `${entry.submission_id} · revision ${entry.revision} · ${entry.submitter}`));
    details.append(node("p", entry.verification));
    if (entry.notes) details.append(node("p", entry.notes));
    for (const limitation of entry.provenance.limitations) details.append(node("p", limitation, "limitation"));
    if (entry.citation) {
      details.append(node("p", entry.citation));
      const citationURL = entry.citation.match(/https?:\/\/[^\s<>]+/u)?.[0]?.replace(/[.,;)]+$/u, "");
      if (citationURL) details.append(link("Read cited paper", citationURL));
    }
    details.append(node("p", "Result records retain full numeric precision. Display rounding does not determine ranks."));
    if (entry.source_type === "reported") {
      details.append(node("p", `Reported ${entry.paper_variant.table}, page ${entry.paper_variant.page}. ${entry.paper_variant.description}. No execution count, seeds or sample standard deviations were reported.`));
      if (entry.experiment) {
        const e = entry.experiment;
        details.append(node("p", `${e.model_family} · ${e.method} · ${e.representation} · ${e.scope}`));
        details.append(disclosure("Reported trait F1 and additional metrics", {trait_f1:e.trait_f1, ...e.additional_reported_metrics}));
        details.append(disclosure("Paper locator and rounding", {source:e.source_document, rounding:e.rounding}));
      }
    } else {
      details.append(node("p", `All ${entry.run_count} explicitly included runs; execution seeds: ${entry.seeds.map((s) => s ?? "unavailable").join(", ")}. Arithmetic mean and sample SD; n=1 has null SD.`));
      details.append(disclosure("Experiment configuration", entry.configuration));
      for (const rid of entry.run_ids) details.append(runDetails(rid, data));
    }
    details.append(disclosure("Immutable submission revision history", data.history[entry.submission_id] || []));
    details.append(disclosure("Full submission metrics and provenance", entry)); return details;
  }

const state = {
  data: null,
  rows: [],
  groups: [],
  groupByKey: new Map(),
  visibleRows: [],
  view: "grouped",
  filters: {
    dataset: "",
    extended: "all",
    adaptation: "all",
    families: new Set(["Encoder", "Decoder"]),
    model: "",
    query: "",
    label: "", task: "", source: "", scope: "main", group: "",
  },
  sort: {
    key: "rank",
    direction: "asc",
  },
};

const elements = {
  body: document.querySelector("#leaderboard-body"),
  labelFilter: document.querySelector("#label-filter"),
  taskFilter: document.querySelector("#task-filter"),
  sourceFilter: document.querySelector("#source-filter"),
  scopeFilter: document.querySelector("#scope-filter"),
  groupFilter: document.querySelector("#group-filter"),
  table: document.querySelector("#leaderboard-table"),
  togetherBody: document.querySelector("#together-leaderboard-body"),
  togetherTable: document.querySelector("#together-leaderboard-table"),
  datasetFilter: document.querySelector("#dataset-filter"),
  extendedButtons: [...document.querySelectorAll("#extended-filter button")],
  adaptationButtons: [...document.querySelectorAll("#adaptation-filter button")],
  viewButtons: [...document.querySelectorAll("#view-mode button")],
  familyInputs: [...document.querySelectorAll(".family-choice input")],
  modelFilter: document.querySelector("#model-filter"),
  searchFilter: document.querySelector("#search-filter"),
  resetButton: document.querySelector("#reset-filters"),
  copyButton: document.querySelector("#copy-results"),
  resultsSummary: document.querySelector("#results-summary"),
  totalConfigurations: document.querySelector("#total-configurations"),
  footerSummary: document.querySelector("#footer-summary"),
  emptyState: document.querySelector("#empty-state"),
  sortButtons: [...document.querySelectorAll(".sort-button")],
  toast: document.querySelector("#toast"),
};

async function getJSON(path) {
  const response = await fetch(path, {cache:"no-cache"});
  if (!response.ok) throw new Error(`Leaderboard data returned ${response.status}`);
  return response.json();
}

function uniqueValues(field) {
  return [...new Set(state.rows.map((row) => row[field]))].sort((left, right) =>
    left.localeCompare(right),
  );
}

function populateSelect(select, values) {
  const fragment = document.createDocumentFragment();
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    fragment.append(option);
  });
  select.replaceChildren(select.options[0], fragment);
}

function groupMatchesFilters(group) {
  return (!state.filters.dataset || group.dataset === state.filters.dataset)
    && (state.filters.extended === "all" || group.extended === state.filters.extended)
    && (state.filters.adaptation === "all" || group.adaptation === state.filters.adaptation)
    && (!state.filters.label || group.label === state.filters.label)
    && (!state.filters.task || group.task === state.filters.task)
    && (!state.filters.group || group.key === state.filters.group);
}

function rowMatchesFilters(row) {
  const entry = row.entry;
  const haystack = [row.model,row.method,row.representation,entry.name,entry.submitter,entry.submission_id].join(" ").toLowerCase();
  return state.filters.families.has(row.model_family)
    && (!state.filters.model || row.model === state.filters.model)
    && (!state.filters.query || haystack.includes(state.filters.query))
    && (!state.filters.source || entry.source_type === state.filters.source)
    && (!state.filters.scope || fieldValue(entry, "scope") === state.filters.scope);
}

function sortedRows(rows) {
  const {key, direction} = state.sort;
  const multiplier = direction === "asc" ? 1 : -1;
  return [...rows].sort((left, right) => {
    const av = left[key], bv = right[key];
    if (av === null || av === undefined) return bv === null || bv === undefined ? 0 : 1;
    if (bv === null || bv === undefined) return -1;
    const comparison = NUMERIC_FIELDS.has(key) ? av - bv : String(av).localeCompare(String(bv));
    return comparison * multiplier || left.model.localeCompare(right.model) || left.sourceIndex - right.sourceIndex;
  });
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function evidenceCell(row) {
  const cell = createElement("td", "source-cell");
  const entry = row.entry;
  cell.append(createElement("span", "evidence-label", entry.source_type === "reported" ? "Reported paper" : "Measured run"));
  if (entry.source_type === "reported") {
    const source = link("Table " + entry.paper_variant.table + " ↗", citationURL(entry));
    source.className = "source-link";
    cell.append(source);
  }
  const button = createElement("button", "source-link details-button", "Details");
  button.type = "button";
  button.setAttribute("aria-label", "Details for " + entry.name);
  button.addEventListener("click", () => {
    const details = entryDetails(entry, state.data);
    details.open = true;
    byId("submission-content").replaceChildren(details);
    byId("submission-dialog").showModal();
  });
  cell.append(button);
  return cell;
}

function citationURL(entry) {
  if (entry.experiment?.source_document?.pdf_page)
    return "https://aclanthology.org/2025.emnlp-main.1137.pdf#page=" + entry.experiment.source_document.pdf_page;
  return entry.citation?.match(/https?:\/\/[^\s<>]+/u)?.[0]?.replace(/[.,;)]+$/u, "") || "";
}

function variantHeader(group, visibleCount) {
  const row = createElement("tr", "variant-row");
  const cell = createElement("th");
  cell.colSpan = 9;
  cell.scope = "rowgroup";

  const identity = createElement("div", "variant-identity");
  const dataset = createElement("span", "variant-dataset", group.dataset);
  const label = createElement("span", "variant-label", group.label);
  identity.append(dataset, label);

  const details = createElement("div", "variant-details");
  details.append(
    createElement("span", "variant-task", TASK_LABELS[group.task] ?? group.task),
    createElement("span", "variant-separator", "·"),
    createElement(
      "span",
      "variant-adaptation",
      ADAPTATION_LABELS[group.adaptation] ?? group.adaptation,
    ),
    createElement("span", "variant-separator", "·"),
    createElement(
      "span",
      group.extended === "true" ? "variant-extended variant-extended--yes" : "variant-extended",
      group.extended === "true" ? "Extended training" : "Standard training",
    ),
  );

  const evidence = ({"documented-paper-variant":"Reported paper","recorded-test-fingerprint":"Measured runs","limited-provenance":"Limited provenance"})[group.comparability] || group.comparability;
  const scoringLabel = ({"perseval.macro-f1.v1":"Macro F1", "perseval.positive-class-f1.v1":"Positive-class F1"})[group.scoring_rule] || group.scoring_rule;
  const partitionLabel = group.comparability === "documented-paper-variant" ? "" : " · " + group.key.slice(-8);
  const provenance = createElement("span", "variant-provenance", evidence + " · " + scoringLabel + partitionLabel);
  provenance.title = group.key + " · " + group.scoring_rule;
  details.append(createElement("span", "variant-separator", "·"), provenance);

  const countText =
    visibleCount === group.rows.length
      ? `${group.rows.length} ${group.rows.length === 1 ? "result" : "results"}`
      : `${visibleCount} of ${group.rows.length} results`;
  const count = createElement("span", "variant-count", countText);

  const content = createElement("div", "variant-content");
  const main = createElement("div", "variant-main");
  main.append(identity, details);
  content.append(main, count);
  cell.append(content);
  row.append(cell);
  return row;
}

function metricCell(row, group, field) {
  const value = row[field], metric = row.entry.metrics[field];
  if (!Number.isFinite(value)) {
    const cell = createElement("td", "metric-cell metric-cell--missing", "—");
    cell.title = "Unavailable";
    return cell;
  }
  const range = group.ranges[field];
  const heat = relativeHeatStyle(value, range, group.rows.length === 1);
  const isBest = value === range.max;
  const cell = createElement("td", `metric-cell${isBest ? " metric-cell--best" : ""}${heat.equal ? " metric-cell--equal" : ""}`);
  cell.style.setProperty("--heat-hue", heat.hue.toFixed(1));
  cell.title = `Full precision: ${value}; scoring rule: ${group.scoring_rule}`;
  const digits = row.entry.experiment?.rounding?.decimal_places ?? 3;
  cell.append(createElement("span", "metric-value", value.toFixed(digits)));
  if (row.entry.source_type === "measured") {
    const text = metric.stddev === null ? `n=${row.entry.run_count}` : `± ${metric.stddev.toFixed(digits)} · n=${row.entry.run_count}`;
    cell.append(createElement("span", "metric-dispersion", text));
    cell.title += "; sample SD: " + (metric.stddev ?? "unavailable");
  }
  if (isBest) cell.append(createElement("span", "sr-only", " (best in this comparison group)"));
  return cell;
}

function resultRow(row, group) {
  const entry = row.entry;
  const tableRow = createElement("tr", "result-row");
  tableRow.dataset.submissionId = entry.submission_id;
  tableRow.dataset.rankingGroup = entry.ranking_group_id;

  const rankCell = createElement("td", "rank-cell");
  const rankBadge = createElement(
    "span",
    `rank-badge${row.rank === 1 ? " rank-badge--first" : ""}`,
    String(entry.rank),
  );
  rankBadge.title = "Backend global F1 rank within the complete comparison group";
  rankCell.append(rankBadge);

  const familyCell = createElement("td");
  const family = createElement(
    "span",
    `family-pill family-pill--${row.model_family.toLowerCase()}`,
    row.model_family,
  );
  familyCell.append(family);

  const modelCell = createElement("td", "model-cell", row.model);
  if (entry.source_type === "measured") modelCell.append(createElement("span", "submission-label", entry.name + " · " + entry.submitter));
  const methodCell = createElement("td", "method-cell", row.method);
  const representationCell = createElement(
    "td",
    "representation-cell",
    row.representation === "none" ? "—" : row.representation,
  );

  const sourceCell = evidenceCell(row);

  tableRow.append(
    rankCell,
    familyCell,
    modelCell,
    methodCell,
    representationCell,
    ...METRIC_FIELDS.map((field) => metricCell(row, group, field)),
    sourceCell,
  );
  return tableRow;
}

function togetherResultRow(row, group) {
  const entry = row.entry;
  const tableRow = createElement("tr", "result-row together-row");
  tableRow.dataset.submissionId = entry.submission_id;
  tableRow.dataset.rankingGroup = entry.ranking_group_id;

  const rankCell = createElement("td", "rank-cell");
  const rankBadge = createElement(
    "span",
    `rank-badge${row.rank === 1 ? " rank-badge--first" : ""}`,
    String(entry.rank),
  );
  rankBadge.title = "Backend global F1 rank within the complete comparison group";
  rankCell.append(rankBadge);

  const datasetCell = createElement("td", "dataset-cell", row.dataset);
  const labelCell = createElement("td", "label-cell", row.label);

  const familyCell = createElement("td");
  const family = createElement(
    "span",
    `family-pill family-pill--${row.model_family.toLowerCase()}`,
    row.model_family,
  );
  familyCell.append(family);

  const modelCell = createElement("td", "model-cell", row.model);
  if (entry.source_type === "measured") modelCell.append(createElement("span", "submission-label", entry.name + " · " + entry.submitter));
  const taskCell = createElement("td", "task-cell", TASK_LABELS[row.task] ?? row.task);
  const adaptationCell = createElement(
    "td",
    "adaptation-cell",
    ADAPTATION_LABELS[row.adaptation] ?? row.adaptation,
  );

  const extendedCell = createElement("td");
  extendedCell.append(
    createElement(
      "span",
      row.extended === "true" ? "extended-pill extended-pill--yes" : "extended-pill",
      row.extended === "true" ? "Yes" : "No",
    ),
  );

  const methodCell = createElement("td", "method-cell", row.method);
  const representationCell = createElement(
    "td",
    "representation-cell",
    row.representation === "none" ? "—" : row.representation,
  );

  const sourceCell = evidenceCell(row);

  tableRow.append(
    rankCell,
    datasetCell,
    labelCell,
    familyCell,
    modelCell,
    taskCell,
    adaptationCell,
    extendedCell,
    methodCell,
    representationCell,
    ...METRIC_FIELDS.map((field) => metricCell(row, group, field)),
    sourceCell,
  );
  return tableRow;
}

function updateSortControls() {
  const sortScope =
    state.view === "grouped" ? "inside each variant" : "across all visible configurations";
  elements.sortButtons.forEach((button) => {
    const active = button.dataset.sort === state.sort.key;
    button.dataset.active = String(active);
    button.dataset.direction = active ? state.sort.direction : "";
    button.setAttribute("aria-pressed", String(active));
    const label = button.textContent.trim();
    button.setAttribute(
      "aria-label",
      active
        ? `${label}, sorted ${state.sort.direction === "asc" ? "ascending" : "descending"} ${sortScope}`
        : `Sort ${sortScope} by ${label}`,
    );
  });
}

function render() {
  const matchingGroups = [];

  state.groups.forEach((group) => {
    if (!groupMatchesFilters(group)) {
      return;
    }

    const matchingRows = group.rows.filter(rowMatchesFilters);
    if (matchingRows.length === 0) {
      return;
    }

    matchingGroups.push({ group, rows: matchingRows });
  });

  const visibleRows = [];
  const groupedFragment = document.createDocumentFragment();
  const togetherFragment = document.createDocumentFragment();

  if (state.view === "grouped") {
    matchingGroups.forEach(({ group, rows }) => {
      const orderedRows = sortedRows(rows);
      groupedFragment.append(variantHeader(group, orderedRows.length));
      orderedRows.forEach((row) => {
        groupedFragment.append(resultRow(row, group));
        visibleRows.push(row);
      });
    });
  } else {
    const orderedRows = sortedRows(matchingGroups.flatMap(({ rows }) => rows));
    orderedRows.forEach((row) => {
      togetherFragment.append(togetherResultRow(row, state.groupByKey.get(row.groupKey)));
      visibleRows.push(row);
    });
  }

  elements.body.replaceChildren(groupedFragment);
  elements.togetherBody.replaceChildren(togetherFragment);
  state.visibleRows = visibleRows;
  elements.table.hidden = visibleRows.length === 0 || state.view !== "grouped";
  elements.togetherTable.hidden = visibleRows.length === 0 || state.view !== "together";
  elements.emptyState.hidden = visibleRows.length !== 0;
  elements.copyButton.disabled = visibleRows.length === 0;

  const visibleGroups = matchingGroups.length;
  const groupWord = visibleGroups === 1 ? "variant" : "variants";
  const viewLabel = state.view === "grouped" ? "grouped view" : "all-together view";
  elements.resultsSummary.textContent =
    `Showing ${visibleRows.length} of ${state.rows.length} submissions across ${visibleGroups} ${groupWord} · ${viewLabel}`;
  byId("empty-title").textContent = state.rows.length ? "No matching submissions" : "No published results yet";
  byId("empty-description").textContent = state.rows.length ? "Try broadening the dataset, model, or search filters." : "Published experiments will appear here.";
  updateSortControls();
}

function resetFilters() {
  state.filters.dataset = "";
  state.filters.extended = "all";
  state.filters.adaptation = "all";
  state.filters.families = new Set(["Encoder", "Decoder"]);
  state.filters.model = "";
  state.filters.query = "";
  Object.assign(state.filters, {label:"",task:"",source:"",scope:"main",group:""});
  for (const key of ["label","task","source","group"]) elements[key + "Filter"].value = "";
  elements.scopeFilter.value = "main";
  state.filters.families.add("Unspecified");
  state.sort =
    state.view === "grouped"
      ? { key: "rank", direction: "asc" }
      : { key: "global_f1", direction: "desc" };

  elements.datasetFilter.value = "";
  elements.modelFilter.value = "";
  elements.searchFilter.value = "";
  elements.extendedButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.value === "all"));
  });
  elements.adaptationButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.value === "all"));
  });
  elements.familyInputs.forEach((input) => {
    input.checked = true;
  });
  render();
}

function bindEvents() {
  for (const key of ["label","task","source","scope","group"]) {
    elements[key + "Filter"].addEventListener("change", () => {
      state.filters[key] = elements[key + "Filter"].value;
      render();
    });
  }
  byId("retry").addEventListener("click", initialize);
  byId("close-details").addEventListener("click", () => byId("submission-dialog").close());
  elements.datasetFilter.addEventListener("change", () => {
    state.filters.dataset = elements.datasetFilter.value;
    render();
  });

  elements.extendedButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.filters.extended = button.dataset.value;
      elements.extendedButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  elements.adaptationButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.filters.adaptation = button.dataset.value;
      elements.adaptationButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  elements.viewButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.value;
      state.sort =
        state.view === "grouped"
          ? { key: "rank", direction: "asc" }
          : { key: "global_f1", direction: "desc" };
      elements.viewButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  elements.familyInputs.forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) {
        state.filters.families.add(input.value);
      } else {
        state.filters.families.delete(input.value);
      }
      render();
    });
  });

  elements.modelFilter.addEventListener("change", () => {
    state.filters.model = elements.modelFilter.value;
    render();
  });

  elements.searchFilter.addEventListener("input", () => {
    state.filters.query = elements.searchFilter.value.trim().toLowerCase();
    render();
  });

  elements.resetButton.addEventListener("click", resetFilters);

  elements.sortButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      if (state.sort.key === key) {
        state.sort.direction = state.sort.direction === "asc" ? "desc" : "asc";
      } else {
        state.sort.key = key;
        state.sort.direction = key === "rank" ? "asc" : "desc";
      }
      render();
    });
  });

  elements.copyButton.addEventListener("click", copyFilteredCsv);
}

async function copyFilteredCsv() {
  const csv = filteredCsv(state.visibleRows);
  try {
    await navigator.clipboard.writeText(csv);
    showToast(`${state.visibleRows.length} filtered rows copied`);
  } catch (_error) {
    const textArea = createElement("textarea", "clipboard-fallback");
    textArea.value = csv;
    textArea.setAttribute("readonly", "");
    document.body.append(textArea);
    textArea.select();
    const copied = typeof document.execCommand === "function" && document.execCommand("copy");
    textArea.remove();
    showToast(copied ? `${state.visibleRows.length} filtered rows copied` : "Copy failed; download the CSV instead.");
  }
}

let toastTimer;
function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 2200);
}

function showLoadError(error) {
  elements.resultsSummary.textContent = "The leaderboard data could not be loaded.";
  elements.table.hidden = true;
  elements.togetherTable.hidden = true;
  elements.emptyState.hidden = true;
  elements.copyButton.disabled = true;
  byId("error-message").textContent = error.message;
  byId("error").hidden = false;
}

async function initialize() {
  byId("error").hidden = true;
  elements.resultsSummary.textContent = "Loading published results…";
  try {
    const data = await getJSON("data/leaderboard.json");
    if (!Array.isArray(data.entries) || !data.runs || !data.history || !data.artifact_metadata)
      throw new Error("Incomplete leaderboard data; rebuild this site.");
    const {rows, groups} = enrichRows(data.entries);
    state.data = data;
    state.rows = rows;
    state.groups = groups;
    state.groupByKey = new Map(groups.map(group => [group.key, group]));
    state.filters.families = new Set(["Encoder", "Decoder", "Unspecified"]);
    elements.totalConfigurations.textContent = rows.length;
    elements.footerSummary.textContent = `${rows.length} submissions · ${uniqueValues("dataset").length} datasets · ${uniqueValues("model").length} models`;
    byId("family-unknown-label").hidden = !rows.some(row => row.model_family === "Unspecified");
    populateSelect(elements.datasetFilter, uniqueValues("dataset"));
    populateSelect(elements.modelFilter, uniqueValues("model"));
    populateSelect(elements.labelFilter, uniqueValues("label"));
    populateSelect(elements.taskFilter, uniqueValues("task"));
    elements.groupFilter.replaceChildren(elements.groupFilter.options[0]);
    for (const group of groups) {
      const option = createElement("option", "", [group.dataset,group.label,TASK_LABELS[group.task] || group.task,ADAPTATION_LABELS[group.adaptation] || group.adaptation,group.comparability,group.key.slice(-8)].join(" · "));
      option.value = group.key;
      elements.groupFilter.append(option);
    }
    byId("history").replaceChildren();
    for (const [sid, revisions] of Object.entries(data.history || {})) {
      const last = revisions[revisions.length - 1];
      const line = node("p", `${last.name} · ${last.source_type} · ${last.status} · ${revisions.length} revision(s) · `);
      line.append(node("span", sid), disclosure("View revisions", revisions));
      byId("history").append(line);
    }
    render();
  } catch (error) {
    showLoadError(error);
  }
}

bindEvents();
initialize();

}
