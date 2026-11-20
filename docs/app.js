const buildings = {
  classroom: "Gore Hall · Classroom",
  residence: "Brown Hall · Residence",
  laboratory: "Building X · Laboratory",
  recreation: "Lil Bob Gym · Recreation",
  library: "Morris Library · Library",
};

const elements = {
  building: document.querySelector("#building-select"),
  historyRange: document.querySelector("#history-range"),
  model: document.querySelector("#model-select"),
  forecastHorizon: document.querySelector("#forecast-horizon"),
  forecastHorizonValue: document.querySelector("#forecast-horizon-value"),
  updated: document.querySelector("#updated-at"),
  latestReading: document.querySelector("#latest-reading"),
  latestReadingTime: document.querySelector("#latest-reading-time"),
  latestSubmission: document.querySelector("#latest-submission"),
  latestSubmissionTime: document.querySelector("#latest-submission-time"),
  nextDeadline: document.querySelector("#next-deadline"),
  deadlineNote: document.querySelector("#deadline-note"),
  errorCount: document.querySelector("#error-count"),
  errorSummary: document.querySelector("#error-summary"),
  systemHealth: document.querySelector(".system-health"),
  errorsCard: document.querySelector("#errors-card"),
  errorsList: document.querySelector("#errors-list"),
  statusDot: document.querySelector("#status-dot"),
  historyChart: document.querySelector("#history-chart"),
  historyEmpty: document.querySelector("#history-empty"),
  forecastChart: document.querySelector("#forecast-chart"),
  forecastEmpty: document.querySelector("#forecast-empty"),
  acceptedRows: document.querySelector("#accepted-rows"),
  evaluationPeriod: document.querySelector("#evaluation-period"),
  overallMae: document.querySelector("#overall-mae"),
  overallRmse: document.querySelector("#overall-rmse"),
  coverage: document.querySelector("#coverage"),
  coverageCount: document.querySelector("#coverage-count"),
  scoredCount: document.querySelector("#scored-count"),
  performanceRows: document.querySelector("#performance-rows"),
  error: document.querySelector("#page-error"),
};

const svgNamespace = "http://www.w3.org/2000/svg";
const dateFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});
const utcFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZone: "UTC",
  timeZoneName: "short",
});
let dashboardData;

function formatTime(value) {
  return dateFormatter.format(new Date(value));
}

function formatUtcTime(value) {
  return utcFormatter.format(new Date(value));
}

function formatDemand(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} kW` : "—";
}

function getModelFamily(model) {
  return model.name.replace(/\s+·\s+\d+-hour$/, "");
}

function formatAge(timestamp) {
  const minutes = Math.max(
    0,
    Math.floor((Date.now() - Date.parse(timestamp)) / 60000),
  );
  if (minutes < 60) return `${minutes} min old`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes
    ? `${hours}h ${remainingMinutes}m old`
    : `${hours}h old`;
}

function setEmptyRows(body, columnCount, message) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = columnCount;
  cell.className = "empty-row";
  cell.textContent = message;
  row.append(cell);
  body.replaceChildren(row);
}

function addTableRow(body, values) {
  const row = document.createElement("tr");
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  body.append(row);
}

function appendSvg(chart, tag, attributes) {
  const node = document.createElementNS(svgNamespace, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, value);
  }
  chart.append(node);
  return node;
}

function renderChart(chart, empty, rows, valueKey, options = {}) {
  chart.replaceChildren();
  empty.hidden = rows.length > 0;
  chart.setAttribute(
    "aria-label",
    `${options.label}, ${rows.length} points`,
  );
  if (rows.length === 0) return;

  const width = 900;
  const height = 320;
  const padding = { top: 16, right: 18, bottom: 38, left: 58 };
  const values = rows.map((row) => row[valueKey]);
  const timeKey = options.timeKey || "timestamp";
  const timestamps = rows.map((row) => Date.parse(row[timeKey]));
  const firstTime = timestamps[0];
  const timeSpan = timestamps[timestamps.length - 1] - firstTime;
  const low = Math.min(...values);
  const high = Math.max(...values);
  const spread = high - low || Math.max(high * 0.1, 1);
  const minValue = Math.max(0, low - spread * 0.15);
  const maxValue = high + spread * 0.15;
  const x = (index) => {
    const position = timeSpan
      ? (timestamps[index] - firstTime) / timeSpan
      : 0.5;
    return padding.left + position * (width - padding.left - padding.right);
  };
  const y = (value) =>
    padding.top +
    ((maxValue - value) / (maxValue - minValue)) *
      (height - padding.top - padding.bottom);

  for (let index = 0; index <= 4; index += 1) {
    const value = minValue + ((maxValue - minValue) * index) / 4;
    const lineY = y(value);
    appendSvg(chart, "line", {
      x1: padding.left,
      x2: width - padding.right,
      y1: lineY,
      y2: lineY,
      class: "grid-line",
    });
    const label = appendSvg(chart, "text", {
      x: padding.left - 10,
      y: lineY + 4,
      "text-anchor": "end",
      class: "axis-label",
    });
    label.textContent = value.toFixed(0);
  }

  if (options.showIntervals) {
    const bandRows = rows.filter(
      (row) =>
        Number.isFinite(row.lower_bound_kw) &&
        Number.isFinite(row.upper_bound_kw),
    );
    if (bandRows.length > 1) {
      const upper = bandRows.map(
        (row) => `${x(rows.indexOf(row))},${y(row.upper_bound_kw)}`,
      );
      const lower = bandRows
        .slice()
        .reverse()
        .map((row) => `${x(rows.indexOf(row))},${y(row.lower_bound_kw)}`);
      appendSvg(chart, "polygon", {
        points: [...upper, ...lower].join(" "),
        class: "confidence-band",
      });
    }
  }

  const linePoints = rows.map((row, index) => `${x(index)},${y(row[valueKey])}`);
  appendSvg(chart, "polyline", {
    points: linePoints.join(" "),
    class: options.showIntervals ? "forecast-line" : "history-line",
  });
  if (options.showIntervals) {
    rows.forEach((row, index) => {
      appendSvg(chart, "circle", {
        cx: x(index),
        cy: y(row[valueKey]),
        r: 4,
        class: "forecast-point",
      });
    });
  } else if (rows.length === 1) {
    appendSvg(chart, "circle", {
      cx: x(0),
      cy: y(rows[0][valueKey]),
      r: 4,
      class: "forecast-point",
    });
  }

  for (const index of [
    ...new Set([0, Math.floor((rows.length - 1) / 2), rows.length - 1]),
  ]) {
    const label = appendSvg(chart, "text", {
      x: x(index),
      y: height - 8,
      "text-anchor":
        index === 0 ? "start" : index === rows.length - 1 ? "end" : "middle",
      class: "axis-label",
    });
    label.textContent = formatTime(
      rows[index][options.timeKey || "timestamp"],
    );
  }

}

function getHistoryRows() {
  const buildingId = elements.building.value;
  const rows = dashboardData.historical_demand
    .filter(
      (row) =>
        row.building_id === buildingId &&
        Number.isFinite(row.demand_kw) &&
        Number.isFinite(Date.parse(row.timestamp)),
    )
    .sort((left, right) => Date.parse(left.timestamp) - Date.parse(right.timestamp));

  if (elements.historyRange.value === "all" || rows.length === 0) return rows;
  const hours = Number(elements.historyRange.value);
  const cutoff = Date.now() - hours * 60 * 60 * 1000;
  return rows.filter((row) => Date.parse(row.timestamp) >= cutoff);
}

function getForecastRows() {
  const models = dashboardData.models.filter(
    (model) => getModelFamily(model) === elements.model.value,
  );
  if (models.length === 0) return [];

  const rows = models
    .flatMap((model) => model.forecasts)
    .filter(
      (row) =>
        row.building_id === elements.building.value &&
        Number.isFinite(row.prediction_kw) &&
        Number.isFinite(Date.parse(row.target_time)),
    )
    .sort(
      (left, right) =>
        Date.parse(left.target_time) - Date.parse(right.target_time),
    );

  if (rows.length === 0) return rows;
  const origin = Date.now();
  const horizonEnd =
    origin + Number(elements.forecastHorizon.value) * 60 * 60 * 1000;
  return rows.filter((row) => {
    const target = Date.parse(row.target_time);
    return target >= origin && target <= horizonEnd;
  });
}

function renderSystemStatus() {
  const status = dashboardData.system_status;
  const latestReading = status.latest_collected_reading;
  if (latestReading && latestReading.timestamp) {
    const demand = latestReading.demand_kw?.[elements.building.value];
    elements.latestReading.textContent = formatDemand(demand);
    elements.latestReadingTime.textContent = `${formatTime(
      latestReading.timestamp,
    )} · ${formatAge(latestReading.timestamp)}`;
    elements.latestReadingTime.classList.toggle(
      "is-stale",
      Date.now() - Date.parse(latestReading.timestamp) > 60 * 60 * 1000,
    );
  } else {
    elements.latestReading.textContent = "—";
    elements.latestReadingTime.textContent = "No reading available";
    elements.latestReadingTime.classList.remove("is-stale");
  }

  const submission = status.latest_accepted_submission;
  if (submission) {
    elements.latestSubmission.textContent = submission.model_version || "Accepted";
    elements.latestSubmissionTime.textContent = submission.accepted_at
      ? formatTime(submission.accepted_at)
      : "Acceptance time unavailable";
  } else {
    elements.latestSubmission.textContent = "—";
    elements.latestSubmissionTime.textContent = "No accepted submission";
  }

  if (status.next_submission_deadline) {
    elements.nextDeadline.textContent = formatUtcTime(
      status.next_submission_deadline,
    );
    elements.deadlineNote.textContent = "Submission window closes · UTC";
  } else {
    elements.nextDeadline.textContent = "—";
    elements.deadlineNote.textContent = "No deadline available";
  }

  const errors = status.errors;
  elements.errorCount.textContent = String(errors.length);
  elements.errorSummary.textContent =
    errors.length === 0
      ? "No recent collection or submission errors"
      : `${errors.length} recent error${errors.length === 1 ? "" : "s"}`;
  elements.statusDot.classList.toggle("has-errors", errors.length > 0);
  elements.systemHealth.classList.toggle("has-errors", errors.length > 0);
  elements.errorsCard.hidden = errors.length === 0;
  elements.errorsList.replaceChildren();
  for (const error of errors.slice(0, 5)) {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = `${error.source || "System"} · ${
      error.timestamp ? formatTime(error.timestamp) : "time unavailable"
    }`;
    const detail = document.createElement("span");
    detail.textContent = error.message || "No error details provided.";
    item.append(heading, detail);
    elements.errorsList.append(item);
  }
}

function renderAcceptedForecasts() {
  const accepted = dashboardData.accepted_forecasts
    .filter((row) => row.building_id === elements.building.value)
    .sort(
      (left, right) =>
        Date.parse(left.target_time) - Date.parse(right.target_time),
    );

  if (accepted.length === 0) {
    setEmptyRows(
      elements.acceptedRows,
      6,
      "No API-accepted forecasts available for this building.",
    );
    return;
  }

  elements.acceptedRows.replaceChildren();
  for (const forecast of accepted.slice(0, 50)) {
    const actual = Number.isFinite(forecast.actual_kw)
      ? formatDemand(forecast.actual_kw)
      : "Pending";
    const absoluteError = Number.isFinite(forecast.actual_kw)
      ? formatDemand(Math.abs(forecast.actual_kw - forecast.prediction_kw))
      : "—";
    addTableRow(elements.acceptedRows, [
      formatUtcTime(forecast.target_time),
      `${forecast.horizon_hours} hours`,
      forecast.model_version,
      formatDemand(forecast.prediction_kw),
      actual,
      absoluteError,
    ]);
  }
}

function renderPerformance() {
  const performance = dashboardData.performance;
  if (!performance) {
    elements.evaluationPeriod.textContent = "PERIOD UNAVAILABLE";
    elements.overallMae.textContent = "—";
    elements.overallRmse.textContent = "—";
    elements.coverage.textContent = "—";
    elements.coverageCount.textContent = "No coverage data";
    elements.scoredCount.textContent = "—";
    setEmptyRows(
      elements.performanceRows,
      5,
      "Performance data will appear here when scored results are published.",
    );
    return;
  }

  const { evaluation_period: period, overall, metrics } = performance;
  elements.evaluationPeriod.textContent =
    period?.start && period?.end
      ? `${formatUtcTime(period.start)} – ${formatUtcTime(period.end)}`
      : "PERIOD UNAVAILABLE";
  elements.overallMae.textContent = formatDemand(overall?.mae_kw);
  elements.overallRmse.textContent = formatDemand(overall?.rmse_kw);
  elements.scoredCount.textContent = Number.isInteger(overall?.scored_predictions)
    ? String(overall.scored_predictions)
    : "—";
  const eligible = overall?.eligible_predictions;
  const submitted = overall?.submitted_predictions;
  if (Number.isFinite(eligible) && Number.isFinite(submitted) && eligible > 0) {
    elements.coverage.textContent = `${((submitted / eligible) * 100).toFixed(1)}%`;
    elements.coverageCount.textContent = `${submitted} submitted / ${eligible} eligible`;
  } else {
    elements.coverage.textContent = "—";
    elements.coverageCount.textContent = "Coverage data unavailable";
  }

  if (metrics.length === 0) {
    setEmptyRows(
      elements.performanceRows,
      5,
      "No per-building evaluation results for this period.",
    );
    return;
  }

  elements.performanceRows.replaceChildren();
  for (const metric of metrics) {
    addTableRow(elements.performanceRows, [
      buildings[metric.building_id] || metric.building_id,
      `${metric.horizon_hours} hours`,
      Number.isFinite(metric.mae_kw) ? metric.mae_kw.toFixed(2) : "—",
      Number.isFinite(metric.rmse_kw) ? metric.rmse_kw.toFixed(2) : "—",
      Number.isInteger(metric.scored_predictions)
        ? String(metric.scored_predictions)
        : "—",
    ]);
  }
}

function render() {
  renderSystemStatus();
  const historicalRows = getHistoryRows();
  renderChart(elements.historyChart, elements.historyEmpty, historicalRows, "demand_kw", {
    label: `${buildings[elements.building.value]} historical demand`,
  });

  const forecastRows = getForecastRows();
  renderChart(
    elements.forecastChart,
    elements.forecastEmpty,
    forecastRows,
    "prediction_kw",
    {
      label: `${buildings[elements.building.value]} model forecast`,
      showIntervals: true,
      timeKey: "target_time",
    },
  );
  renderAcceptedForecasts();
  renderPerformance();
  elements.forecastHorizonValue.textContent = `${elements.forecastHorizon.value} ${
    elements.forecastHorizon.value === "1" ? "hour" : "hours"
  }`;
  elements.updated.textContent = dashboardData.updated_at
    ? formatTime(dashboardData.updated_at)
    : "Waiting for data";
}

function validateData(data) {
  if (
    !data ||
    !Array.isArray(data.historical_demand) ||
    !Array.isArray(data.models) ||
    !Array.isArray(data.accepted_forecasts) ||
    !data.system_status ||
    !Array.isArray(data.system_status.errors) ||
    (data.performance !== null &&
      (!data.performance || !Array.isArray(data.performance.metrics)))
  ) {
    throw new Error("Forecast data does not match the dashboard data format.");
  }
  for (const model of data.models) {
    if (
      typeof model.id !== "string" ||
      typeof model.name !== "string" ||
      !Array.isArray(model.forecasts)
    ) {
      throw new Error("Each model must include an id, name, and forecasts array.");
    }
  }
}

async function loadData() {
  elements.error.hidden = true;
  try {
    const response = await fetch("./data/forecasts.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Dashboard data request failed (${response.status}).`);
    }
    const data = await response.json();
    validateData(data);
    dashboardData = data;

    elements.model.replaceChildren();
    const modelFamilies = [
      ...new Set(data.models.map((model) => getModelFamily(model))),
    ];
    for (const family of modelFamilies) {
      const option = document.createElement("option");
      option.value = family;
      option.textContent = family;
      elements.model.append(option);
    }
    elements.model.disabled = elements.model.options.length === 0;
    if (elements.model.options.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No model data";
      elements.model.append(option);
    }
    render();
  } catch (error) {
    elements.updated.textContent = "Data unavailable";
    elements.error.textContent = `Unable to load dashboard data: ${error.message}`;
    elements.error.hidden = false;
  }
}

for (const control of [elements.building, elements.historyRange, elements.model]) {
  control.addEventListener("change", () => {
    if (dashboardData) render();
  });
}
elements.forecastHorizon.addEventListener("input", () => {
  if (dashboardData) render();
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) loadData();
});
window.setInterval(() => {
  if (!document.hidden) loadData();
}, 5 * 60 * 1000);
loadData();
