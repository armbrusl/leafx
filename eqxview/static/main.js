const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const vizEl = document.getElementById("viz");
const dropOverlayEl = document.getElementById("dropOverlay");
const processingOverlayEl = document.getElementById("processingOverlay");
const processingTextEl = document.getElementById("processingText");
const inspectorEl = document.getElementById("inspector");
const inspectorDockEl = document.getElementById("inspectorDock");
const infoBtn = document.getElementById("infoBtn");
const hudDetailsEl = document.getElementById("hudDetails");
const hudEl = document.querySelector(".hud");
const collapseBtn = document.getElementById("collapseBtn");
const defaultBtn = document.getElementById("defaultBtn");
const shapesBtn = document.getElementById("shapesBtn");
const themeBtn = document.getElementById("themeBtn");
const heatmapBtn = document.getElementById("heatmapBtn");

let dragDepth = 0;
let currentGraph = null;
let collapsedGroups = new Set();
let allCollapsed = true;
let currentTransform = null;
let currentInputShapeValue = "";
let currentInputDtype = "";
let currentInputHeatmap = null;
let currentInputFile = null;
let currentInputFlatValues = null;
let currentOutputShape = null;
let currentOutputDtype = "";
let currentOutputHeatmap = null;
let currentOutputFlatValues = null;
let currentOutputStats = null;
let currentNodeActivations = {};
let expandedHeatmaps = new Set();
let selectiveParamHeatmaps = new Set();
let tensorProjectionModes = new Map();
let showShapes = false;
let showHeatmaps = false;
let helpOpen = false;
let themeMode = "auto";

const systemLightQuery =
  typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-color-scheme: light)")
    : null;

/* ── Utilities ──────────────────────────────────────────── */

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
  if (infoBtn) infoBtn.classList.toggle("has-alert", isError && !helpOpen);
}

function formatNumber(n) {
  return new Intl.NumberFormat().format(n ?? 0);
}

function renderSummary(summary) {
  summaryEl.innerHTML = [
    `parameters: ${formatNumber(summary.total_parameters)}`,
    `bytes: ${formatNumber(summary.total_bytes)} (~${summary.total_megabytes} MB)`,
    `leaf_count: ${formatNumber(summary.leaf_count)}`,
  ]
    .map((line) => `<div>${line}</div>`)
    .join("");
}

function shapeStr(shape) {
  if (!shape || !shape.length) return "?";
  return "[" + shape.join(", ") + "]";
}

function compactShapeStr(shape, maxChars = 16) {
  if (!shape || !shape.length) return "?";
  const raw = shape.join("x");
  if (raw.length <= maxChars) return raw;
  if (shape.length === 1) return raw.slice(0, Math.max(1, maxChars - 3)) + "...";
  let label = `${shape.length}D ${shape.slice(0, 2).join("x")}x...x${shape.at(-1)}`;
  if (label.length <= maxChars) return label;
  label = `${shape.length}D ${shape[0]}x...x${shape.at(-1)}`;
  if (label.length <= maxChars) return label;
  return `${shape.length}D ${shape[0]}...${shape.at(-1)}`;
}

function product(values) {
  if (!values?.length) return 1;
  return values.reduce((acc, value) => acc * value, 1);
}

function squeezeShape(shape) {
  if (!Array.isArray(shape) || !shape.length) return shape;
  const squeezed = shape.filter((dim) => dim !== 1);
  return squeezed.length ? squeezed : [1];
}

function identityAxisOrder(n) {
  return Array.from({ length: n }, (_, idx) => idx);
}

function axisOrderKey(order) {
  return Array.isArray(order) ? order.join(",") : "";
}

function parseAxisOrderKey(key) {
  if (!key) return null;
  const order = key.split(",").map((value) => Number(value));
  return order.every((value) => Number.isInteger(value)) ? order : null;
}

function generateAxisOrders(shape) {
  const squeezedShape = squeezeShape(shape);
  const n = Array.isArray(squeezedShape) ? squeezedShape.length : 0;
  if (n <= 2) return [];
  const base = identityAxisOrder(n);
  if (n > 4) return [base];
  const orders = [];
  function permute(prefix, remaining) {
    if (!remaining.length) {
      orders.push(prefix);
      return;
    }
    for (let idx = 0; idx < remaining.length; idx++) {
      permute([...prefix, remaining[idx]], [...remaining.slice(0, idx), ...remaining.slice(idx + 1)]);
    }
  }
  permute([], base);
  orders.sort((left, right) => {
    const leftKey = axisOrderKey(left);
    const rightKey = axisOrderKey(right);
    if (leftKey === axisOrderKey(base)) return -1;
    if (rightKey === axisOrderKey(base)) return 1;
    return leftKey.localeCompare(rightKey);
  });
  return orders;
}

function getAxisOrderForTensor(tensorId, shape) {
  const orders = generateAxisOrders(shape);
  if (!orders.length) return null;
  const saved = parseAxisOrderKey(tensorProjectionModes.get(tensorId));
  if (saved) {
    const match = orders.find((order) => axisOrderKey(order) === axisOrderKey(saved));
    if (match) return match;
  }
  return orders[0];
}

function setAxisOrderForTensor(tensorId, order) {
  if (!tensorId || !Array.isArray(order)) return;
  tensorProjectionModes.set(tensorId, axisOrderKey(order));
}

function tensorProjection(shape, values2d = null, axisOrder = null) {
  const squeezedShape = squeezeShape(shape);
  if (Array.isArray(squeezedShape) && squeezedShape.length) {
    const order = Array.isArray(axisOrder) && axisOrder.length === squeezedShape.length
      ? axisOrder
      : identityAxisOrder(squeezedShape.length);
    const orderedShape = order.map((idx) => squeezedShape[idx]);
    if (orderedShape.length === 1) {
      return {
        rows: 1,
        cols: orderedShape[0],
        rowShape: [],
        colShape: [orderedShape[0]],
        displayShape: squeezedShape,
        orderedShape,
      };
    }
    if (orderedShape.length === 2) {
      return {
        rows: orderedShape[0],
        cols: orderedShape[1],
        rowShape: [orderedShape[0]],
        colShape: [orderedShape[1]],
        displayShape: squeezedShape,
        orderedShape,
        projected: false,
      };
    }
    const outerShape = orderedShape.slice(0, -2);
    const innerRows = orderedShape[orderedShape.length - 2];
    const innerCols = orderedShape[orderedShape.length - 1];
    const outerSplit = Math.max(1, Math.ceil(outerShape.length / 2));
    const blockRowShape = outerShape.slice(0, outerSplit);
    const blockColShape = outerShape.slice(outerSplit);
    const blockRows = product(blockRowShape) || 1;
    const blockCols = product(blockColShape) || 1;
    return {
      rows: blockRows * innerRows,
      cols: blockCols * innerCols,
      rowShape: [blockRows, innerRows],
      colShape: [blockCols, innerCols],
      blockRowShape,
      blockColShape,
      blockRows,
      blockCols,
      innerRows,
      innerCols,
      displayShape: squeezedShape,
      orderedShape,
      axisOrder: order,
      projected: true,
    };
  }
  return {
    rows: values2d?.length || 1,
    cols: values2d?.[0]?.length || 1,
    rowShape: values2d?.length ? [values2d.length] : [],
    colShape: values2d?.[0]?.length ? [values2d[0].length] : [],
    displayShape: values2d?.length ? [values2d.length, values2d?.[0]?.length || 1] : null,
    projected: false,
  };
}

function tensorProjectionLabel(shape, axisOrder = null) {
  const squeezedShape = squeezeShape(shape);
  if (!Array.isArray(squeezedShape) || squeezedShape.length <= 2) return null;
  const projection = tensorProjection(squeezedShape, null, axisOrder);
  if (!projection.projected) return null;
  const rowBlocks = projection.blockRowShape?.length ? projection.blockRowShape.join("x") : "1";
  const colBlocks = projection.blockColShape?.length ? projection.blockColShape.join("x") : "1";
  return `${rowBlocks} by ${colBlocks} blocks of ${projection.innerRows}x${projection.innerCols}`;
}

function inflateTensorIndex(originalShape, squeezedIndex) {
  if (!Array.isArray(originalShape) || !originalShape.length) return squeezedIndex;
  const out = [];
  let cursor = 0;
  for (const dim of originalShape) {
    if (dim === 1) {
      out.push(0);
    } else {
      out.push(squeezedIndex[cursor] ?? 0);
      cursor += 1;
    }
  }
  return out;
}

function unflattenIndex(index, dims) {
  if (!dims?.length) return [];
  const out = new Array(dims.length);
  let rest = index;
  for (let i = dims.length - 1; i >= 0; i--) {
    const dim = Math.max(1, dims[i]);
    out[i] = rest % dim;
    rest = Math.floor(rest / dim);
  }
  return out;
}

function projectFlatValues(shape, flatValues, axisOrder) {
  const squeezedShape = squeezeShape(shape);
  if (!Array.isArray(squeezedShape) || !squeezedShape.length) {
    return [[flatValues?.[0] ?? 0]];
  }
  const total = product(squeezedShape);
  if (!Array.isArray(flatValues) || flatValues.length < total) return null;

  const projection = tensorProjection(shape, null, axisOrder);
  const rank = squeezedShape.length;
  const order = Array.isArray(axisOrder) && axisOrder.length === rank
    ? axisOrder
    : identityAxisOrder(rank);
  const strides = new Array(rank);
  let stride = 1;
  for (let idx = rank - 1; idx >= 0; idx--) {
    strides[idx] = stride;
    stride *= squeezedShape[idx];
  }

  const values = Array.from({ length: projection.rows }, () => new Array(projection.cols).fill(0));
  for (let row = 0; row < projection.rows; row++) {
    const blockRowIndex = projection.projected ? Math.floor(row / projection.innerRows) : 0;
    const innerRowIndex = projection.projected ? row % projection.innerRows : row;
    const blockRowCoords = projection.projected ? unflattenIndex(blockRowIndex, projection.blockRowShape) : [];
    for (let col = 0; col < projection.cols; col++) {
      const blockColIndex = projection.projected ? Math.floor(col / projection.innerCols) : 0;
      const innerColIndex = projection.projected ? col % projection.innerCols : col;
      const blockColCoords = projection.projected ? unflattenIndex(blockColIndex, projection.blockColShape) : [];
      const reorderedCoords = projection.projected
        ? [...blockRowCoords, ...blockColCoords, innerRowIndex, innerColIndex]
        : [row, col];
      const originalCoords = new Array(rank).fill(0);
      for (let pos = 0; pos < order.length; pos++) {
        originalCoords[order[pos]] = reorderedCoords[pos] ?? 0;
      }
      let flatIndex = 0;
      for (let idx = 0; idx < rank; idx++) {
        flatIndex += originalCoords[idx] * strides[idx];
      }
      values[row][col] = flatValues[flatIndex] ?? 0;
    }
  }
  return values;
}

function tensorSupportsProjectionCycling(shape, flatValues) {
  return Array.isArray(flatValues) && generateAxisOrders(shape).length > 1;
}

function tensorAxisOrder(tensorId, shape) {
  return getAxisOrderForTensor(tensorId, shape);
}

function tensorDisplayValues(datum) {
  if (!datum) return null;
  if (!tensorSupportsProjectionCycling(datum.shapeInfo?.shape, datum.flatValues)) {
    return datum.values;
  }
  const axisOrder = tensorAxisOrder(datum.node?.id, datum.shapeInfo?.shape);
  return projectFlatValues(datum.shapeInfo?.shape, datum.flatValues, axisOrder) || datum.values;
}

function cycleTensorProjection(tensorId, shape, direction = 1) {
  const orders = generateAxisOrders(shape);
  if (orders.length <= 1) return null;
  const current = getAxisOrderForTensor(tensorId, shape);
  const currentIdx = orders.findIndex((order) => axisOrderKey(order) === axisOrderKey(current));
  const nextIdx = currentIdx < 0
    ? 0
    : (currentIdx + direction + orders.length) % orders.length;
  const next = orders[nextIdx];
  setAxisOrderForTensor(tensorId, next);
  return next;
}

function tensorIndexLabel(shape, row, col, axisOrder = null) {
  if (!Array.isArray(shape) || !shape.length) {
    return row === 0 ? `[${col}]` : `[${row}, ${col}]`;
  }
  const squeezedShape = squeezeShape(shape);
  if (squeezedShape.length === 1) {
    return `[${inflateTensorIndex(shape, [col]).join(", ")}]`;
  }
  if (squeezedShape.length === 2) {
    return `[${inflateTensorIndex(shape, [row, col]).join(", ")}]`;
  }
  const projection = tensorProjection(shape, null, axisOrder);
  const blockRowIndex = Math.floor(row / projection.innerRows);
  const innerRowIndex = row % projection.innerRows;
  const blockColIndex = Math.floor(col / projection.innerCols);
  const innerColIndex = col % projection.innerCols;
  const squeezedIndex = [
    ...unflattenIndex(blockRowIndex, projection.blockRowShape),
    ...unflattenIndex(blockColIndex, projection.blockColShape),
    innerRowIndex,
    innerColIndex,
  ];
  const fullIndex = inflateTensorIndex(shape, squeezedIndex);
  return `[${fullIndex.join(", ")}]`;
}

function projectionGuideLines(shape, values2d = null, axisOrder = null) {
  const projection = tensorProjection(shape, values2d, axisOrder);
  if (!projection.projected || projection.blockRows <= 1 && projection.blockCols <= 1) {
    return null;
  }
  const rowLines = [];
  const colLines = [];
  for (let block = 1; block < projection.blockRows; block++) {
    rowLines.push(block * projection.innerRows);
  }
  for (let block = 1; block < projection.blockCols; block++) {
    colLines.push(block * projection.innerCols);
  }
  return {
    rowLines,
    colLines,
    label: tensorProjectionLabel(shape, axisOrder),
  };
}

function drawProjectionGuides(ctx, shape, values2d, cellW = 1, cellH = 1, axisOrder = null) {
  const guides = projectionGuideLines(shape, values2d, axisOrder);
  if (!guides) return;
  const rows = values2d?.length || 0;
  const cols = values2d?.[0]?.length || 0;
  if (!rows || !cols) return;

  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.95)";

  for (const rowLine of guides.rowLines) {
    const y = Math.max(0, Math.round(rowLine * cellH) - 1);
    ctx.fillRect(0, y, cols * cellW, 1);
  }
  for (const colLine of guides.colLines) {
    const x = Math.max(0, Math.round(colLine * cellW) - 1);
    ctx.fillRect(x, 0, 1, rows * cellH);
  }
  ctx.restore();
}

function bytesStr(count) {
  if (count < 1024) return `${count} B`;
  if (count < 1024 * 1024) return `${(count / 1024).toFixed(1)} KB`;
  return `${(count / (1024 * 1024)).toFixed(2)} MB`;
}

function parseInputShape() {
  const raw = currentInputShapeValue.trim();
  if (!raw) return null;
  const parts = raw.split(/[,x×\s]+/).map(Number).filter((n) => n > 0 && Number.isFinite(n));
  return parts.length ? parts : null;
}

function downsample2d(values2d, maxSize = 64) {
  if (!values2d || !values2d.length) return null;
  const rows = values2d.length;
  const cols = values2d[0].length;
  const rIdx = rows > maxSize
    ? Array.from({ length: maxSize }, (_, i) => Math.floor((i * (rows - 1)) / (maxSize - 1)))
    : Array.from({ length: rows }, (_, i) => i);
  const cIdx = cols > maxSize
    ? Array.from({ length: maxSize }, (_, i) => Math.floor((i * (cols - 1)) / (maxSize - 1)))
    : Array.from({ length: cols }, (_, i) => i);
  return rIdx.map((r) => cIdx.map((c) => values2d[r][c]));
}

function parseNpyHeader(buffer) {
  const dv = new DataView(buffer);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < magic.length; i++) {
    if (dv.getUint8(i) !== magic[i]) throw new Error("Not a valid .npy file");
  }

  const major = dv.getUint8(6);
  let headerLen;
  let headerOffset;
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    headerOffset = 10;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    headerOffset = 12;
  } else {
    throw new Error(`Unsupported .npy version ${major}`);
  }

  const headerTxt = new TextDecoder("latin1").decode(
    new Uint8Array(buffer, headerOffset, headerLen)
  );

  const descr = /'descr'\s*:\s*'([^']+)'/.exec(headerTxt)?.[1];
  const fortranOrder = /'fortran_order'\s*:\s*(True|False)/.exec(headerTxt)?.[1] === "True";
  const shapeRaw = /'shape'\s*:\s*\(([^\)]*)\)/.exec(headerTxt)?.[1] || "";
  const shape = shapeRaw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length)
    .map((s) => Number(s));

  if (!descr || !shape.length || shape.some((n) => !Number.isFinite(n) || n <= 0)) {
    throw new Error("Failed to parse .npy header");
  }

  return {
    descr,
    fortranOrder,
    shape,
    dataOffset: headerOffset + headerLen,
  };
}

function npyDescrToDtype(descr) {
  const m = /^([<>|=]?)([a-zA-Z])(\d+)$/.exec(descr);
  if (!m) return descr;
  const kind = m[2];
  const size = Number(m[3]);
  if (kind === "f") return `float${size * 8}`;
  if (kind === "i") return `int${size * 8}`;
  if (kind === "u") return `uint${size * 8}`;
  if (kind === "b") return "bool";
  return descr;
}

function npyReadAsFloat32(buffer, dataOffset, descr, count) {
  const m = /^([<>|=]?)([a-zA-Z])(\d+)$/.exec(descr);
  if (!m) throw new Error(`Unsupported dtype descriptor: ${descr}`);
  const endian = m[1] || "=";
  const kind = m[2];
  const size = Number(m[3]);
  const little = endian !== ">";
  const dv = new DataView(buffer, dataOffset);
  const out = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const off = i * size;
    let v;
    if (kind === "f") {
      if (size === 8) v = dv.getFloat64(off, little);
      else if (size === 4) v = dv.getFloat32(off, little);
      else if (size === 2) {
        // Approximate float16 by reading as uint16 and expanding roughly.
        const h = dv.getUint16(off, little);
        const s = (h & 0x8000) ? -1 : 1;
        const e = (h >> 10) & 0x1f;
        const f = h & 0x03ff;
        if (e === 0) v = s * (f / 1024) * 2 ** (-14);
        else if (e === 0x1f) v = f ? NaN : s * Infinity;
        else v = s * (1 + f / 1024) * 2 ** (e - 15);
      } else throw new Error(`Unsupported float width: ${size}`);
    } else if (kind === "i") {
      if (size === 8) v = Number(dv.getBigInt64(off, little));
      else if (size === 4) v = dv.getInt32(off, little);
      else if (size === 2) v = dv.getInt16(off, little);
      else if (size === 1) v = dv.getInt8(off);
      else throw new Error(`Unsupported int width: ${size}`);
    } else if (kind === "u") {
      if (size === 8) v = Number(dv.getBigUint64(off, little));
      else if (size === 4) v = dv.getUint32(off, little);
      else if (size === 2) v = dv.getUint16(off, little);
      else if (size === 1) v = dv.getUint8(off);
      else throw new Error(`Unsupported uint width: ${size}`);
    } else if (kind === "b") {
      v = dv.getUint8(off) ? 1 : 0;
    } else {
      throw new Error(`Unsupported dtype kind: ${kind}`);
    }
    out[i] = Number.isFinite(v) ? v : 0;
  }

  return out;
}

function makeStrides(shape, fortranOrder) {
  const n = shape.length;
  const strides = new Array(n).fill(1);
  if (fortranOrder) {
    for (let i = 1; i < n; i++) strides[i] = strides[i - 1] * shape[i - 1];
  } else {
    for (let i = n - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

function flattenIndex(indices, strides) {
  let idx = 0;
  for (let i = 0; i < indices.length; i++) idx += indices[i] * strides[i];
  return idx;
}

function npyToHeatmap2d(data, shape, fortranOrder) {
  if (!shape.length) return null;
  const ndim = shape.length;
  const strides = makeStrides(shape, fortranOrder);

  if (ndim === 1) {
    return [Array.from({ length: shape[0] }, (_, c) => data[c])];
  }

  if (ndim === 2) {
    const [rows, cols] = shape;
    const out = Array.from({ length: rows }, () => new Array(cols));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out[r][c] = data[flattenIndex([r, c], strides)];
      }
    }
    return out;
  }

  if (ndim === 3 && shape[2] <= 4) {
    // HWC image: average channels for a scalar heatmap.
    const [rows, cols, ch] = shape;
    const out = Array.from({ length: rows }, () => new Array(cols));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let s = 0;
        for (let k = 0; k < ch; k++) s += data[flattenIndex([r, c, k], strides)];
        out[r][c] = s / ch;
      }
    }
    return out;
  }

  // Generic fallback: first slice of leading dimensions, show last 2 dims.
  const rows = shape[shape.length - 2];
  const cols = shape[shape.length - 1];
  const lead = new Array(shape.length - 2).fill(0);
  const out = Array.from({ length: rows }, () => new Array(cols));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[r][c] = data[flattenIndex([...lead, r, c], strides)];
    }
  }
  return out;
}

async function loadInputFromNpy(file) {
  const buf = await file.arrayBuffer();
  const hdr = parseNpyHeader(buf);
  const count = hdr.shape.reduce((a, b) => a * b, 1);
  const vals = npyReadAsFloat32(buf, hdr.dataOffset, hdr.descr, count);
  const hm = npyToHeatmap2d(vals, hdr.shape, hdr.fortranOrder);

  currentInputShapeValue = hdr.shape.join(", ");
  currentInputDtype = npyDescrToDtype(hdr.descr);
  currentInputFlatValues = vals;
  currentInputHeatmap = downsample2d(hm, 64);
  currentInputFile = file;

  if (currentGraph) {
    const runResult = await runCurrentModelWithInput();
    await renderOperationGraph(currentGraph);
    if (!runResult.ok) {
      setStatus(runResult.error || "Failed to run model on input.", true);
      return;
    }
    setStatus(`Loaded INPUT tensor ${file.name} and updated OUTPUT (${currentOutputDtype}, ${shapeStr(currentOutputShape)}).`);
    return;
  }
  setStatus(`Loaded INPUT tensor ${file.name} (${currentInputDtype}, [${hdr.shape.join(", ")}]).`);
}

function clearCurrentOutput() {
  currentOutputShape = null;
  currentOutputDtype = "";
  currentOutputHeatmap = null;
  currentOutputFlatValues = null;
  currentOutputStats = null;
  currentNodeActivations = {};
  expandedHeatmaps.delete("__OUTPUT__");
}

async function runCurrentModelWithInput() {
  if (!currentInputFile) {
    clearCurrentOutput();
    return { ok: false, skipped: true };
  }

  const formData = new FormData();
  formData.append("input_file", currentInputFile);

  try {
    const res = await fetch("/api/run-current-model", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to run model on input.");
    }

    const data = await res.json();
    currentOutputShape = Array.isArray(data.shape) ? data.shape : null;
    currentOutputDtype = data.dtype || "";
    currentOutputHeatmap = Array.isArray(data.heatmap) ? data.heatmap : null;
    currentOutputFlatValues = Array.isArray(data.flat_values) ? data.flat_values : null;
    currentOutputStats = data.stats || null;
    currentNodeActivations = data.activations && typeof data.activations === "object"
      ? data.activations
      : {};
    return { ok: true };
  } catch (err) {
    clearCurrentOutput();
    return { ok: false, error: err.message || "Failed to run model on input." };
  }
}

/* ── Shape inference ────────────────────────────────────── */

function itemsizeForDtype(dtype) {
  const d = String(dtype || "").toLowerCase();
  if (d === "bool") return 1;
  const m = /(\d+)$/.exec(d);
  if (m) return Math.max(1, Number(m[1]) / 8);
  return 4;
}

function inferEdgeShapes(graph) {
  const inputShape = parseInputShape();
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const outputShapeOf = new Map();
  const edgeShapes = new Map();

  if (inputShape) outputShapeOf.set("__INPUT__", inputShape);

  const paramEdgesByTarget = new Map();
  graph.edges.forEach((e, i) => {
    if (e.kind === "parameter") {
      const arr = paramEdgesByTarget.get(e.target) || [];
      arr.push({ edge: e, idx: i });
      paramEdgesByTarget.set(e.target, arr);
    }
  });

  graph.edges.forEach((e, idx) => {
    if (e.kind === "parameter") {
      if (e.shape) {
        const count = e.shape.reduce((a, b) => a * b, 1);
        edgeShapes.set(idx, { shape: e.shape, dtype: e.dtype, bytes: count * itemsizeForDtype(e.dtype) });
      }
      return;
    }

    let flowShape = null;
    const node = nodeById.get(e.target);
    const label = node?.label || "";

    if (!e.source) {
      flowShape = inputShape;
    } else {
      flowShape = outputShapeOf.get(e.source) || null;
    }

    if (label.startsWith("MatMul") && flowShape) {
      const params = paramEdgesByTarget.get(e.target) || [];
      const weightEdge = params.find((p) => p.edge.label?.startsWith("weight"));
      if (weightEdge?.edge.shape?.length >= 2) {
        const outFeatures = weightEdge.edge.shape[0];
        const outShape = [...flowShape.slice(0, -1), outFeatures];
        outputShapeOf.set(e.target, outShape);
      } else {
        outputShapeOf.set(e.target, flowShape);
      }
    } else if (label.startsWith("Layer") && flowShape) {
      const params = paramEdgesByTarget.get(e.target) || [];
      const lastWeight = [...params].reverse().find((p) => p.edge.label?.startsWith("weight"));
      if (lastWeight?.edge.shape?.length >= 2) {
        outputShapeOf.set(e.target, [...flowShape.slice(0, -1), lastWeight.edge.shape[0]]);
      } else {
        outputShapeOf.set(e.target, flowShape);
      }
    } else if (flowShape) {
      outputShapeOf.set(e.target, flowShape);
    }

    if (flowShape) {
      const count = flowShape.reduce((a, b) => a * b, 1);
      edgeShapes.set(idx, { shape: flowShape, dtype: "float32", bytes: count * 4 });
    }
  });

  return { edgeShapes, nodeShapes: outputShapeOf };
}

function captureFinalNodeIds(graph) {
  const finalByCapture = new Map();
  graph.nodes.forEach((node) => {
    if (!node.capture_path) return;
    const prev = finalByCapture.get(node.capture_path);
    if (!prev || node.depth > prev.depth) finalByCapture.set(node.capture_path, node);
  });
  return new Map(Array.from(finalByCapture.entries()).map(([cp, node]) => [cp, node.id]));
}

/* ── Collapsing ─────────────────────────────────────────── */

function getGroups(graph) {
  const groups = new Map();
  graph.nodes.forEach((n) => {
    if (n.group) {
      const arr = groups.get(n.group) || [];
      arr.push(n);
      groups.set(n.group, arr);
    }
  });
  return groups;
}

function collapseGraph(graph, collapsedPrefixes) {
  if (!collapsedPrefixes.size) return graph;

  const groupMap = getGroups(graph);

  // For each group id, find the longest matching collapsed prefix
  function matchingPrefix(gid) {
    if (!gid) return null;
    let best = null;
    for (const prefix of collapsedPrefixes) {
      if (gid === prefix || gid.startsWith(prefix + "/")) {
        if (!best || prefix.length > best.length) best = prefix;
      }
    }
    return best;
  }

  // Collect nodes per matching prefix
  const prefixNodes = new Map();
  groupMap.forEach((nodes, gid) => {
    const prefix = matchingPrefix(gid);
    if (!prefix) return;
    const arr = prefixNodes.get(prefix) || [];
    arr.push(...nodes);
    prefixNodes.set(prefix, arr);
  });

  const collapsedNodeIds = new Set();
  const groupReplace = new Map();
  const newNodes = [];
  const newEdges = [];

  prefixNodes.forEach((nodes, prefix) => {
    const sorted = [...nodes].sort((a, b) => a.depth - b.depth);
    const first = sorted[0];
    const collapsedId = `collapsed:${prefix}`;

    newNodes.push({
      id: collapsedId,
      label: prefix,
      depth: first.depth,
      type: "layer",
      path: prefix,
      group: prefix,
      _collapsed: true,
      _collapsedPrefix: prefix,
      _memberIds: sorted.map((n) => n.id),
    });

    sorted.forEach((n) => {
      collapsedNodeIds.add(n.id);
      groupReplace.set(n.id, collapsedId);
    });
  });

  graph.nodes.forEach((n) => {
    if (!collapsedNodeIds.has(n.id)) newNodes.push(n);
  });

  const seenEdges = new Set();
  graph.edges.forEach((e) => {
    const src = e.source ? (groupReplace.get(e.source) || e.source) : e.source;
    const tgt = groupReplace.get(e.target) || e.target;

    if (src && collapsedNodeIds.has(e.source) && collapsedNodeIds.has(e.target)) {
      if (groupReplace.get(e.source) === groupReplace.get(e.target)) return;
    }

    if (e.kind === "parameter" && collapsedNodeIds.has(e.target)) {
      newEdges.push({ ...e, target: tgt });
      return;
    }

    const key = `${e.kind}:${src}:${tgt}:${e.label}`;
    if (seenEdges.has(key)) return;
    seenEdges.add(key);

    newEdges.push({ ...e, source: src, target: tgt });
  });

  return { nodes: newNodes, edges: newEdges };
}

// Get top-level path segments (the first path component after stripping model root)
function getTopLevelPrefixes(graph) {
  const groups = getGroups(graph);
  const tops = new Set();
  groups.forEach((_, gid) => {
    const first = gid.split("/")[0];
    if (first) tops.add(first);
  });
  return tops;
}

// Given a prefix being expanded, find the unique child-level prefixes
function getChildPrefixes(graph, parentPrefix) {
  const groups = getGroups(graph);
  const children = new Set();
  const depth = parentPrefix.split("/").length;
  groups.forEach((_, gid) => {
    if (gid === parentPrefix || gid.startsWith(parentPrefix + "/")) {
      const parts = gid.split("/");
      if (parts.length > depth) {
        children.add(parts.slice(0, depth + 1).join("/"));
      }
    }
  });
  return children;
}

/* ── Graph fallback ─────────────────────────────────────── */

function graphFromTree(tree) {
  if (!tree || typeof tree !== "object") return { nodes: [], edges: [] };
  const nodes = [];
  const edges = [];
  const rootId = "op:root";
  nodes.push({ id: rootId, label: tree.name || "Model", depth: 0 });

  function walk(node, parentId, depth, path) {
    const children = Array.isArray(node?.children) ? node.children : [];
    children.filter((c) => c && c.kind !== "leaf").forEach((child, idx) => {
      const childName = child.name || "node";
      const childId = `op:${path}/${childName}`;
      nodes.push({ id: childId, label: childName, depth });
      edges.push({ source: parentId, target: childId, kind: "flow", label: idx > 0 ? "activation" : "input" });
      walk(child, childId, depth + 1, `${path}/${childName}`);
    });
    children.filter((c) => c && c.kind === "leaf").forEach((leaf) => {
      edges.push({
        source: null, target: parentId, kind: "parameter",
        label: `${leaf.name || "tensor"} ${Array.isArray(leaf.shape) ? "[" + leaf.shape.join("x") + "]" : ""}`.trim(),
        shape: leaf.shape, dtype: leaf.dtype,
      });
    });
  }
  walk(tree, rootId, 1, "root");
  return { nodes, edges };
}

/* ── Viz helpers ────────────────────────────────────────── */

function clearViz() { vizEl.innerHTML = ""; }

function hideInspector() {
  if (inspectorDockEl) inspectorDockEl.classList.add("hidden");
  inspectorEl.classList.add("hidden");
  inspectorEl.innerHTML = "";
}

function showInspector(title, rows) {
  if (inspectorDockEl) inspectorDockEl.classList.remove("hidden");
  inspectorEl.classList.remove("hidden");
  inspectorEl.innerHTML = [
    `<div class="title">${title}</div>`,
    ...rows.map((r) => `<div class="row"><span class="key">${r.key}:</span> ${r.value}</div>`),
  ].join("");
}

function updateInspectorDockPosition() {
  if (!inspectorDockEl || !hudEl) return;
  const top = hudEl.offsetTop + hudEl.offsetHeight + 8;
  inspectorDockEl.style.top = `${top}px`;
}

function setHelpOpen(open) {
  helpOpen = !!open;
  if (hudDetailsEl) hudDetailsEl.classList.toggle("hidden", !helpOpen);
  if (infoBtn) {
    infoBtn.classList.toggle("is-active", helpOpen);
    infoBtn.classList.toggle("has-alert", false);
    infoBtn.setAttribute("aria-expanded", helpOpen ? "true" : "false");
  }
  updateInspectorDockPosition();
}

if (infoBtn) {
  infoBtn.addEventListener("click", () => setHelpOpen(!helpOpen));
}

setHelpOpen(false);
window.addEventListener("resize", updateInspectorDockPosition);
updateInspectorDockPosition();

function compactEdgeLabel(label) {
  if (!label) return "";
  return String(label).split(" [")[0].trim();
}

function edgePathFromPoints(points) {
  if (!points || points.length < 2) return "";
  if (points.length === 2) {
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
  }
  const gen = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
  return gen(points);
}

/* ── Heatmap rendering ──────────────────────────────────── */

function valuesStats(values2d) {
  if (!values2d || !values2d.length) return null;
  let min = Infinity, max = -Infinity, sum = 0, count = 0;
  for (const row of values2d) for (const v of row) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    count++;
  }
  const mean = sum / count;
  let sq = 0;
  for (const row of values2d) for (const v of row) sq += (v - mean) ** 2;
  return { min, max, mean, std: Math.sqrt(sq / count), count };
}

function resolveHeatRange(values2d, rangeOverride = null) {
  if (rangeOverride && Number.isFinite(rangeOverride.vmin) && Number.isFinite(rangeOverride.vmax)) {
    return { vmin: rangeOverride.vmin, vmax: rangeOverride.vmax };
  }
  let vmin = Infinity, vmax = -Infinity;
  for (let r = 0; r < values2d.length; r++) {
    for (let c = 0; c < values2d[r].length; c++) {
      const v = values2d[r][c];
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
  }
  return { vmin, vmax };
}

function tensorDisplayDims(shape, values2d) {
  if (values2d?.length) {
    return {
      rows: values2d.length,
      cols: values2d[0]?.length || 1,
    };
  }
  const projection = tensorProjection(shape, values2d);
  return { rows: projection.rows, cols: projection.cols };
}

function drawHeatmap(canvas, values2d, rangeOverride = null, shape = null, axisOrder = null) {
  if (!values2d || !values2d.length) return;
  const rows = values2d.length;
  const cols = values2d[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);

  const { vmin, vmax } = resolveHeatRange(values2d, rangeOverride);

  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const v = values2d[r][c];
      const col = heatColorRGB(v, vmin, vmax);
      const i = (r * cols + c) * 4;
      img.data[i]     = col.r;
      img.data[i + 1] = col.g;
      img.data[i + 2] = col.b;
      img.data[i + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
  drawProjectionGuides(ctx, shape, values2d, 1, 1, axisOrder);
}

function heatmapBaseCellPx(rows, cols) {
  // Small/medium tensors get more visual resolution; larger ones stay compact.
  const cells = rows * cols;
  const maxDim = Math.max(rows, cols);
  if (cells <= 1024 && maxDim <= 48) return 4;
  if (cells <= 2500 && maxDim <= 64) return 3;
  return 2;
}

function heatmapThumbSize(values2d, shape = null) {
  if (!values2d?.length && !shape?.length) return null;
  const dims = tensorDisplayDims(shape, values2d);
  const rows = dims.rows;
  const cols = dims.cols;
  // Fixed 2 px per cell so tensor size directly determines display size.
  const ppc = 2;
  const maxW = 560;
  const maxH = 400;

  // Keep pseudo-1D bias strips thin.
  if (rows === 1) {
    return {
      w: Math.max(24, Math.min(maxW, cols * ppc)),
      h: 4,
    };
  }

  const rawW = cols * ppc;
  const rawH = rows * ppc;
  // Scale uniformly: fit within caps, then ensure minimums — always preserving
  // the data aspect ratio so projections with extreme shapes aren't distorted.
  const fitDown = Math.min(1, maxW / rawW, maxH / rawH);
  const fitUp = Math.max(1, 12 / (rawW * fitDown), 12 / (rawH * fitDown));
  const scale = fitDown * fitUp;
  return {
    w: Math.max(12, Math.round(rawW * scale)),
    h: Math.max(12, Math.round(rawH * scale)),
  };
}

function scaleThumbSize(thumb, factor = 1) {
  if (!thumb) return null;
  return {
    w: Math.min(920, Math.max(24, Math.round(thumb.w * factor))),
    h: Math.min(720, Math.max(4, Math.round(thumb.h * factor))),
  };
}

function tensorThumbSize(datum) {
  if (!datum) return null;
  return heatmapThumbSize(tensorDisplayValues(datum), datum.shapeInfo?.shape);
}

function heatColorRGB(v, vmin, vmax) {
  const range = vmax - vmin || 1;
  const tLinear = Math.max(0, Math.min(1, (v - vmin) / range));
  const centered = tLinear * 2 - 1;
  const emphasized = Math.sign(centered) * Math.abs(centered) ** 0.45;
  const mag = Math.abs(emphasized);
  const base = { r: 252, g: 252, b: 252 };
  if (emphasized < 0) {
    const neg = { r: 20, g: 110, b: 255 };
    const r = Math.round(base.r + (neg.r - base.r) * mag);
    const g = Math.round(base.g + (neg.g - base.g) * mag);
    const b = Math.round(base.b + (neg.b - base.b) * mag);
    const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    return { r, g, b, luminance };
  }
  const pos = { r: 255, g: 40, b: 40 };
  const r = Math.round(base.r + (pos.r - base.r) * mag);
  const g = Math.round(base.g + (pos.g - base.g) * mag);
  const b = Math.round(base.b + (pos.b - base.b) * mag);
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return { r, g, b, luminance };
}

function formatHeatValue(v) {
  const a = Math.abs(v);
  if (a >= 1000 || (a > 0 && a < 0.001)) return v.toExponential(2);
  if (a >= 10) return v.toFixed(2);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

function colorForGroup(groupId) {
  const text = String(groupId || "");
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
  }
  const isLight = document.documentElement.classList.contains("light");

  if (isLight) {
    const hue = hash % 360;
    return {
      stroke: `hsl(${hue} 44% 48% / 0.7)`,
      fill: `hsl(${hue} 36% 58% / 0.045)`,
    };
  }

  // Curated high-contrast hues for dark backgrounds.
  const darkHues = [
    10, 28, 46, 68, 92, 118, 145, 172, 198, 224, 248, 272, 298, 324, 346,
  ];
  const hue = darkHues[hash % darkHues.length];
  return {
    stroke: `hsl(${hue} 82% 72% / 0.96)`,
    fill: `hsl(${hue} 78% 60% / 0.14)`,
  };
}

function pointOnRectBoundaryToward(rectCenter, rectSize, toward) {
  const cx = rectCenter.x;
  const cy = rectCenter.y;
  const hw = Math.max(1, rectSize.w / 2);
  const hh = Math.max(1, rectSize.h / 2);
  const vx = toward.x - cx;
  const vy = toward.y - cy;

  if (vx === 0 && vy === 0) {
    return { x: cx + hw, y: cy };
  }

  const sx = Math.abs(vx) / hw;
  const sy = Math.abs(vy) / hh;
  if (sx >= sy) {
    const x = cx + Math.sign(vx || 1) * hw;
    const y = cy + (Math.abs(vx) < 1e-6 ? 0 : vy * (hw / Math.abs(vx)));
    return { x, y };
  }

  const y = cy + Math.sign(vy || 1) * hh;
  const x = cx + (Math.abs(vy) < 1e-6 ? 0 : vx * (hh / Math.abs(vy)));
  return { x, y };
}

function snapRouteEndpointsToRenderedNodes(routePoints, sourceNode, targetNode) {
  if (!routePoints || routePoints.length < 2) return routePoints;
  const pts = routePoints.map((p) => ({ x: p.x, y: p.y }));

  if (sourceNode) {
    pts[0] = pointOnRectBoundaryToward(
      { x: sourceNode.x, y: sourceNode.y },
      { w: sourceNode.w, h: sourceNode.h },
      pts[1]
    );
  }

  if (targetNode) {
    const n = pts.length;
    pts[n - 1] = pointOnRectBoundaryToward(
      { x: targetNode.x, y: targetNode.y },
      { w: targetNode.w, h: targetNode.h },
      pts[n - 2]
    );
  }

  return pts;
}

/* ── Main render ────────────────────────────────────────── */

async function renderOperationGraph(rawGraph) {
  /* Preserve zoom / pan across re-renders */
  const prevSvg = vizEl.querySelector("svg");
  if (prevSvg) { try { currentTransform = d3.zoomTransform(prevSvg); } catch (_) {} }

  clearViz();

  const collapsed = collapseGraph(rawGraph, collapsedGroups);
  const graph = { nodes: [...collapsed.nodes], edges: [...collapsed.edges] };

  /* Inject INPUT node */
  graph.nodes.unshift({ id: "__INPUT__", label: "Input", depth: -1, type: "input" });
  graph.edges = graph.edges.map(e =>
    e.kind === "flow" && !e.source ? { ...e, source: "__INPUT__" } : e
  );

  /* Inject OUTPUT node */
  const _fSrc = new Set(graph.edges.filter(e => e.kind === "flow").map(e => e.source));
  const _fTgt = new Set(graph.edges.filter(e => e.kind === "flow").map(e => e.target));
  const _terms = graph.nodes.filter(n => n.id !== "__INPUT__" && _fTgt.has(n.id) && !_fSrc.has(n.id));
  if (_terms.length) {
    const lastN = _terms.reduce((a, b) => (a.depth > b.depth ? a : b));
    graph.nodes.push({ id: "__OUTPUT__", label: "Output", depth: lastN.depth + 1, type: "output" });
    graph.edges.push({ source: lastN.id, target: "__OUTPUT__", label: "output", kind: "flow", shape: null, dtype: null });
  }

  const { edgeShapes, nodeShapes } = inferEdgeShapes(graph);
  const anyParamHeatmapVisible = showHeatmaps || selectiveParamHeatmaps.size > 0;
  const globalHeatRange = anyParamHeatmapVisible ? { vmin: -1, vmax: 1 } : null;
  const finalCaptureNodeById = captureFinalNodeIds(graph);

  function paramHeatmapVisible(paramId) {
    return showHeatmaps || selectiveParamHeatmaps.has(paramId);
  }

  function activationForNode(node) {
    if (!node?.capture_path) return null;
    if (finalCaptureNodeById.get(node.capture_path) !== node.id) return null;
    return currentNodeActivations[node.capture_path] || null;
  }

  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    setStatus("Invalid graph payload from server.", true);
    return;
  }

  const width = vizEl.clientWidth || 800;
  const height = vizEl.clientHeight || 600;

  const svg = d3
    .select(vizEl)
    .append("svg")
    .attr("viewBox", [0, 0, width, height])
    .attr("preserveAspectRatio", "xMidYMid meet");

  const container = svg.append("g");
  const zoomAwareOverlays = [];
  const updateZoomAwareOverlays = (k) => {
    zoomAwareOverlays.forEach((fn) => fn(k));
  };
  // Use a very wide range so wheel zoom feels effectively unbounded.
  const zoomBehavior = d3.zoom().scaleExtent([1e-6, 1e6]).on("zoom", (event) => {
    container.attr("transform", event.transform);
    currentTransform = event.transform;
    updateZoomAwareOverlays(event.transform.k);
  });
  svg.call(zoomBehavior);
  // Reserve double-click for app gestures (expand/collapse all), not zoom.
  svg.on("dblclick.zoom", null);
  if (currentTransform) svg.call(zoomBehavior.transform, currentTransform);

  const opNodeW = 78;
  const opNodeH = 24;
  const layerNodeH = 28;
  const paramNodeW = 56;
  const paramNodeH = 18;

  // Compute collapsed node width based on label length
  function collapsedNodeW(n) {
    const label = n.label || "";
    return Math.max(86, label.length * 7.5 + 20);
  }

  function heatmapRoutePad(w, h) {
    const base = Math.max(w || 0, h || 0);
    return Math.max(16, Math.min(72, Math.round(base * 0.18)));
  }

  const inputShape = parseInputShape();
  const inputDatum = {
    node: { id: "__INPUT__", label: "input" },
    shapeInfo: inputShape
      ? {
        shape: inputShape,
        dtype: currentInputDtype || "float32",
        bytes: inputShape.reduce((a, b) => a * b, 1) * itemsizeForDtype(currentInputDtype || "float32"),
      }
      : null,
    values: currentInputHeatmap,
    flatValues: Array.isArray(currentInputFlatValues) ? currentInputFlatValues : null,
  };
  const inputThumbBase = (showHeatmaps && currentInputHeatmap)
    ? tensorThumbSize(inputDatum)
    : null;
  const inputThumb = expandedHeatmaps.has("__INPUT__")
    ? scaleThumbSize(inputThumbBase, 2.5)
    : inputThumbBase;
  const inputNodeW = inputThumb ? inputThumb.w : 150;
  const inputNodeH = inputThumb ? inputThumb.h : 40;
  const outputDatum = {
    node: { id: "__OUTPUT__", label: "output", type: "output" },
    shapeInfo: currentOutputShape
      ? {
        shape: currentOutputShape,
        dtype: currentOutputDtype || null,
        bytes: null,
      }
      : null,
    values: currentOutputHeatmap,
    flatValues: Array.isArray(currentOutputFlatValues) ? currentOutputFlatValues : null,
  };
  const outputThumbBase = (showHeatmaps && currentOutputHeatmap)
    ? tensorThumbSize(outputDatum)
    : null;
  const outputThumb = expandedHeatmaps.has("__OUTPUT__")
    ? scaleThumbSize(outputThumbBase, 2.5)
    : outputThumbBase;
  const outputNodeW = outputThumb ? outputThumb.w : 150;
  const outputNodeH = outputThumb ? outputThumb.h : 40;

  const renderSizeById = new Map();
  const elkChildren = graph.nodes.map((n) => {
    const activation = activationForNode(n);
    const activationDatum = activation?.heatmap?.length
      ? {
        node: { id: n.id, label: n.label, type: n.type },
        shapeInfo: {
          shape: activation.shape,
          dtype: activation.dtype,
          bytes: activation.bytes,
        },
        values: activation.heatmap,
        flatValues: Array.isArray(activation.flat_values) ? activation.flat_values : null,
      }
      : null;
    const activationThumb = showHeatmaps ? tensorThumbSize(activationDatum) : null;
    const hasActHeatmap = !!activationThumb;
    const w =
      n.type === "input"
        ? inputNodeW
        : n.type === "output"
          ? outputNodeW
          : hasActHeatmap
            ? activationThumb?.w || opNodeW
          : n._collapsed
            ? collapsedNodeW(n)
            : opNodeW;
    const h =
      n.type === "input"
        ? inputNodeH
        : n.type === "output"
          ? outputNodeH
          : hasActHeatmap
            ? activationThumb?.h || opNodeH
          : n._collapsed
            ? layerNodeH
            : opNodeH;

    // Give heatmap nodes extra invisible layout margin so ELK routes edges around them.
    const routePad =
      showHeatmaps && (
        (n.type === "input" && currentInputHeatmap) ||
        (n.type === "output" && currentOutputHeatmap) ||
        hasActHeatmap
      )
        ? heatmapRoutePad(w, h)
        : 0;

    renderSizeById.set(n.id, { w, h });
    return { id: n.id, width: w + routePad * 2, height: h + routePad * 2 };
  });

  const sourceOverride = new Map();
  const paramSizeByEdge = new Map();
  graph.edges.forEach((edge, idx) => {
    if (!edge.source) {
      const synthId = `param-src-${idx}`;
      if (paramHeatmapVisible(synthId) && edge.values && edge.values.length) {
        const baseThumb = tensorThumbSize({
          node: { id: synthId, label: compactEdgeLabel(edge.label), type: "parameter" },
          shapeInfo: edge.shape
            ? {
              shape: edge.shape,
              dtype: edge.dtype,
              bytes: edge.shape.reduce((a, b) => a * b, 1) * itemsizeForDtype(edge.dtype),
            }
            : null,
          values: edge.values || null,
          flatValues: Array.isArray(edge.flat_values) ? edge.flat_values : null,
        });
        const thumb = expandedHeatmaps.has(synthId) ? scaleThumbSize(baseThumb, 2.5) : baseThumb;
        if (thumb) {
          const routePad = heatmapRoutePad(thumb.w, thumb.h);
          elkChildren.push({ id: synthId, width: thumb.w + routePad * 2, height: thumb.h + routePad * 2 });
          renderSizeById.set(synthId, { w: thumb.w, h: thumb.h });
          paramSizeByEdge.set(idx, thumb);
        } else {
          elkChildren.push({ id: synthId, width: paramNodeW, height: paramNodeH });
          renderSizeById.set(synthId, { w: paramNodeW, h: paramNodeH });
        }
      } else {
        elkChildren.push({ id: synthId, width: paramNodeW, height: paramNodeH });
        renderSizeById.set(synthId, { w: paramNodeW, h: paramNodeH });
      }
      sourceOverride.set(idx, synthId);
    }
  });

  const elkEdges = graph.edges.map((e, i) => ({
    id: `edge-${i}`,
    sources: [sourceOverride.get(i) || e.source],
    targets: [e.target],
  }));

  const elk = new ELK();
  let laidOut;
  try {
    laidOut = await elk.layout({
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.layered.spacing.nodeNodeBetweenLayers": "92",
        "elk.spacing.nodeNode": "35",
        "elk.spacing.edgeNode": "24",
        "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
        "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
        "elk.edgeRouting": "ORTHOGONAL",
      },
      children: elkChildren,
      edges: elkEdges,
    });
  } catch (err) {
    setStatus("Layout failed: " + (err.message || err), true);
    return;
  }

  const elkNodes = laidOut.children || [];
  if (!elkNodes.length) { setStatus("Empty layout.", true); return; }

  const xs = elkNodes.map((n) => n.x + n.width / 2);
  const ys = elkNodes.map((n) => n.y + n.height / 2);
  const offX = width / 2 - (Math.min(...xs) + Math.max(...xs)) / 2;
  const offY = height / 2 - (Math.min(...ys) + Math.max(...ys)) / 2;

  const elkNodeById = new Map(elkNodes.map((n) => [n.id, n]));
  const nodePos = new Map();

  graph.nodes.forEach((n) => {
    const en = elkNodeById.get(n.id);
    if (en) {
      const renderSize = renderSizeById.get(n.id);
      nodePos.set(n.id, {
        x: en.x + en.width / 2 + offX,
        y: en.y + en.height / 2 + offY,
        w: renderSize?.w || en.width, h: renderSize?.h || en.height,
        node: n, isParam: false,
      });
    }
  });

  const paramNodesData = [];
  sourceOverride.forEach((synthId, edgeIdx) => {
    const en = elkNodeById.get(synthId);
    if (en) {
      const edge = graph.edges[edgeIdx];
      const si = edgeShapes.get(edgeIdx);
      const psize = paramSizeByEdge.get(edgeIdx);
      const info = {
        x: en.x + en.width / 2 + offX,
        y: en.y + en.height / 2 + offY,
        w: psize?.w || en.width, h: psize?.h || en.height,
        node: { id: synthId, label: compactEdgeLabel(edge.label), type: "parameter" },
        isParam: true,
        shapeInfo: si || (edge.shape ? { shape: edge.shape, dtype: edge.dtype, bytes: (edge.shape.reduce((a,b)=>a*b,1)) * itemsizeForDtype(edge.dtype) } : null),
        values: edge.values || null,
        flatValues: Array.isArray(edge.flat_values) ? edge.flat_values : null,
      };
      nodePos.set(synthId, info);
      paramNodesData.push(info);
    }
  });

  const elkEdgeById = new Map((laidOut.edges || []).map((e) => [e.id, e]));

  const links = graph.edges.map((edge, idx) => {
    const elkEdge = elkEdgeById.get(`edge-${idx}`);
    const src = sourceOverride.get(idx) || edge.source;

    let routePoints = null;
    if (elkEdge?.sections?.length) {
      const sec = elkEdge.sections[0];
      routePoints = [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint]
        .map((p) => ({ x: p.x + offX, y: p.y + offY }));

      const srcNode = nodePos.get(src) || null;
      const dstNode = nodePos.get(edge.target) || null;
      routePoints = snapRouteEndpointsToRenderedNodes(routePoints, srcNode, dstNode);
    }

    const shapeInfo = edgeShapes.get(idx) || null;

    return { ...edge, _src: src, routePoints, edgeIndex: idx, shapeInfo };
  });

  /* ── Render ─────────────────────────────────────────── */
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const hullG = container.append("g").attr("class", "group-hulls");
  const linkG = container.append("g").attr("class", "links");
  const nodeG = container.append("g").attr("class", "nodes");

  // Draw colored hulls around visible group subtrees (expanded + collapsed members).
  const opNodesForHull = Array.from(nodePos.values()).filter(
    (d) => !d.isParam && d.node.type !== "input" && d.node.type !== "output" && d.node.group
  );
  const byGroup = new Map();
  opNodesForHull.forEach((d) => {
    const parts = String(d.node.group).split("/").filter(Boolean);
    for (let i = 1; i <= parts.length; i++) {
      const prefix = parts.slice(0, i).join("/");
      const arr = byGroup.get(prefix) || [];
      arr.push(d);
      byGroup.set(prefix, arr);
    }
  });
  const hullPadX = 18;
  const hullPadY = 14;
  const maxHullLevel = Math.max(
    1,
    ...Array.from(byGroup.keys()).map((g) => String(g).split("/").length)
  );
  const hullData = Array.from(byGroup.entries())
    .map(([group, members]) => {
      const label = group.includes("/") ? group.split("/").at(-1) : group;
      const level = group.split("/").length;
      // If two levels cover the same members, this makes nested levels still visible.
      const levelExtra = (maxHullLevel - level) * 8;
      const padX = hullPadX + levelExtra;
      const padY = hullPadY + Math.round(levelExtra * 0.7);
      const x0 = Math.min(...members.map((m) => m.x - m.w / 2)) - padX;
      const x1 = Math.max(...members.map((m) => m.x + m.w / 2)) + padX;
      const y0 = Math.min(...members.map((m) => m.y - m.h / 2)) - padY;
      const y1 = Math.max(...members.map((m) => m.y + m.h / 2)) + padY;
      return { group, x: x0, y: y0, w: x1 - x0, h: y1 - y0, label, level };
    })
    // Draw outer groups first so deeper levels remain visible on top.
    .sort((a, b) => a.level - b.level);

  const hullSel = hullG.selectAll("g.group-hull").data(hullData).enter()
    .append("g")
    .attr("class", "group-hull");

  hullSel.append("rect")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y)
    .attr("width", (d) => d.w)
    .attr("height", (d) => d.h)
    .attr("rx", 8)
    .style("fill", (d) => colorForGroup(d.group).fill);

  hullSel.append("text")
    .attr("x", (d) => (d.level % 2 === 1 ? d.x + 8 : d.x + d.w - 8))
    .attr("y", (d) => d.y - 4)
    .attr("text-anchor", (d) => (d.level % 2 === 1 ? "start" : "end"))
    .text((d) => d.label)
    .style("fill", (d) => colorForGroup(d.group).stroke);

  /* edges */
  linkG.selectAll("path").data(links).enter().append("path")
    .attr("class", (d) => `link ${d.kind === "parameter" ? "link-param" : "link-flow"}`)
    .attr("d", (d) => {
      if (d.routePoints?.length >= 2) return edgePathFromPoints(d.routePoints);
      const s = nodePos.get(d._src);
      const t = nodePos.get(d.target);
      if (s && t) return `M ${s.x} ${s.y} L ${t.x} ${t.y}`;
      return "";
    })
    .on("mouseenter", (_, d) => {
      const srcLabel = d._src ? (nodeById.get(d._src)?.label || d._src) : "external";
      const dstLabel = nodeById.get(d.target)?.label || d.target;
      const rows = [
        { key: "name", value: d.label || "-" },
        { key: "kind", value: d.kind || "-" },
        { key: "from", value: srcLabel },
        { key: "to", value: dstLabel },
      ];
      if (d.shapeInfo?.shape) rows.push({ key: "shape", value: shapeStr(d.shapeInfo.shape) });
      if (d.shapeInfo?.dtype) rows.push({ key: "dtype", value: d.shapeInfo.dtype });
      if (d.shapeInfo?.bytes != null) rows.push({ key: "size", value: bytesStr(d.shapeInfo.bytes) });
      showInspector("Edge", rows);
    })
    .on("mouseleave", () => hideInspector());

  function nodeLabel(d) {
    if (!showShapes) return d.node.label;
    const s = nodeShapes.get(d.node.id);
    return s ? compactShapeStr(s) : d.node.label;
  }

  function paramLabel(d) {
    if (!showShapes) return d.node.label;
    return d.shapeInfo?.shape ? compactShapeStr(d.shapeInfo.shape) : d.node.label;
  }

  function parameterInspectorRows(d, extraRows = []) {
    const rows = [{ key: "name", value: d.node.label || "-" }];
    if (d.shapeInfo?.shape) rows.push({ key: "shape", value: shapeStr(d.shapeInfo.shape) });
    const axisOrder = tensorAxisOrder(d.node?.id, d.shapeInfo?.shape);
    const projection = tensorProjectionLabel(d.shapeInfo?.shape, axisOrder);
    if (projection) rows.push({ key: "view", value: projection });
    if (d.shapeInfo?.dtype) rows.push({ key: "dtype", value: d.shapeInfo.dtype });
    if (d.shapeInfo?.bytes != null) rows.push({ key: "size", value: bytesStr(d.shapeInfo.bytes) });
    const visibleValues = tensorDisplayValues(d);
    if (paramHeatmapVisible(d.node.id) && visibleValues) {
      const st = valuesStats(visibleValues);
      if (st) {
        rows.push({ key: "min", value: st.min.toFixed(4) });
        rows.push({ key: "max", value: st.max.toFixed(4) });
        rows.push({ key: "mean", value: st.mean.toFixed(4) });
        rows.push({ key: "std", value: st.std.toFixed(4) });
      }
    }
    return [...extraRows, ...rows];
  }

  /* operation nodes */
  const opNodes = Array.from(nodePos.values()).filter((d) => !d.isParam && d.node.type !== "input" && d.node.type !== "output");
  const gOp = nodeG.selectAll("g.node").data(opNodes).enter()
    .append("g")
    .attr("class", (d) => {
      let cls = d.node._collapsed ? "node layer-node" : "node";
      if (d.node.capture_path) cls += " capture-node";
      return cls;
    })
    .attr("transform", (d) => `translate(${d.x},${d.y})`)
    .style("cursor", (d) => d.node._collapsed || d.node.group ? "pointer" : "default");

  gOp.on("mouseenter", (_, d) => {
    const activation = activationForNode(d.node);
    const rows = [
      { key: "name", value: d.node.label || "-" },
      { key: "type", value: d.node.type || "operation" },
    ];
    if (d.node._collapsed && d.node._memberIds) {
      rows.push({ key: "ops", value: d.node._memberIds.length + " operations" });
    }
    if (d.node.path) rows.push({ key: "path", value: d.node.path });
    if (d.node.capture_name) rows.push({ key: "capture", value: d.node.capture_name });
    rows.push({ key: "group", value: d.node.group || "-" });
    if (activation?.shape) rows.push({ key: "shape", value: shapeStr(activation.shape) });
    const projection = tensorProjectionLabel(activation?.shape);
    if (projection) rows.push({ key: "view", value: projection });
    if (activation?.dtype) rows.push({ key: "dtype", value: activation.dtype });
    if (activation?.bytes != null) rows.push({ key: "size", value: bytesStr(activation.bytes) });
    if (showHeatmaps && activation?.stats) {
      rows.push({ key: "min", value: activation.stats.min.toFixed(4) });
      rows.push({ key: "max", value: activation.stats.max.toFixed(4) });
      rows.push({ key: "mean", value: activation.stats.mean.toFixed(4) });
      rows.push({ key: "std", value: activation.stats.std.toFixed(4) });
    }
    showInspector(d.node._collapsed ? "Layer" : "Node", rows);
  }).on("mouseleave", () => hideInspector());

  function expandOneLevel(d) {
    if (!d.node._collapsed || !d.node._collapsedPrefix) return;
    // Expand one level: remove this prefix, add child-level prefixes.
    const prefix = d.node._collapsedPrefix;
    collapsedGroups.delete(prefix);
    const children = getChildPrefixes(currentGraph, prefix);
    if (children.size > 0) {
      // Children exist: keep them collapsed so expansion is one level at a time.
      children.forEach((cp) => collapsedGroups.add(cp));
    }
    allCollapsed = false;
    updateCollapseButton();
    renderOperationGraph(currentGraph);
  }

  function toggleTerminalParameterHeatmaps(nodeId) {
    const paramIds = [];
    graph.edges.forEach((edge, idx) => {
      if (edge.kind === "parameter" && edge.target === nodeId) {
        paramIds.push(`param-src-${idx}`);
      }
    });
    if (!paramIds.length) return false;

    const hasHidden = paramIds.some((id) => !selectiveParamHeatmaps.has(id));
    if (hasHidden) {
      paramIds.forEach((id) => selectiveParamHeatmaps.add(id));
    } else {
      paramIds.forEach((id) => {
        selectiveParamHeatmaps.delete(id);
        expandedHeatmaps.delete(id);
      });
    }
    renderOperationGraph(currentGraph);
    return true;
  }

  function collapseGroup(d) {
    let targetPrefix = null;

    if (d.node._collapsed && d.node._collapsedPrefix) {
      const parts = String(d.node._collapsedPrefix).split("/").filter(Boolean);
      targetPrefix = parts.length > 1 ? parts.slice(0, -1).join("/") : d.node._collapsedPrefix;
    } else if (!d.node._collapsed && d.node.group) {
      targetPrefix = d.node.group;
    }

    if (!targetPrefix) return;

    // Replace any descendant collapsed markers with the requested collapsed prefix.
    for (const p of Array.from(collapsedGroups)) {
      if (p === targetPrefix || p.startsWith(targetPrefix + "/")) {
        collapsedGroups.delete(p);
      }
    }
    collapsedGroups.add(targetPrefix);
    renderOperationGraph(currentGraph);
  }

  gOp.on("click", (event, d) => {
    // Left click expands collapsed nodes. It does not resize heatmaps.
    if (event.button !== 0) return;
    if (d.node._collapsed) {
      const childPrefixes = d.node._collapsedPrefix
        ? getChildPrefixes(currentGraph, d.node._collapsedPrefix)
        : new Set();
      // Terminal collapsed node: final click toggles attached parameter heatmaps.
      if (!childPrefixes.size && toggleTerminalParameterHeatmaps(d.node.id)) return;
      expandOneLevel(d);
      return;
    }
    const activation = activationForNode(d.node);
    if (showHeatmaps && activation?.heatmap?.length) {
      const flatVals = Array.isArray(activation.flat_values) ? activation.flat_values : null;
      if (tensorSupportsProjectionCycling(activation.shape, flatVals)) {
        cycleTensorProjection(d.node.id, activation.shape, 1);
        renderOperationGraph(currentGraph);
      }
      return;
    }

    // Final step for expanded terminal op nodes: toggle their attached
    // parameter heatmaps if no activation heatmap is available.
    toggleTerminalParameterHeatmaps(d.node.id);
  });

  gOp.on("contextmenu", (event, d) => {
    // Right click collapses this group (or parent level for collapsed placeholders).
    event.preventDefault();
    const activation = activationForNode(d.node);
    if (showHeatmaps && activation?.heatmap?.length) {
      const flatVals = Array.isArray(activation.flat_values) ? activation.flat_values : null;
      if (tensorSupportsProjectionCycling(activation.shape, flatVals)) {
        cycleTensorProjection(d.node.id, activation.shape, -1);
        renderOperationGraph(currentGraph);
        return;
      }
    }
    collapseGroup(d);
  });

  gOp.each(function (d) {
    const g = d3.select(this);
    const activation = activationForNode(d.node);

    if (showHeatmaps && activation?.heatmap?.length) {
      const clipId = `op-hm-clip-${d.node.id.replace(/[^a-zA-Z0-9]/g, "_")}`;
      g.append("defs").append("clipPath").attr("id", clipId)
        .append("rect")
        .attr("x", -d.w / 2)
        .attr("y", -d.h / 2)
        .attr("width", d.w)
        .attr("height", d.h)
        .attr("rx", 0);

      const fo = g.append("foreignObject")
        .attr("x", -d.w / 2)
        .attr("y", -d.h / 2)
        .attr("width", d.w)
        .attr("height", d.h)
        .attr("clip-path", `url(#${clipId})`)
        .style("pointer-events", "none");

      const activationDatum = {
        node: { id: d.node.id, label: d.node.label, type: d.node.type },
        shapeInfo: {
          shape: activation.shape,
          dtype: activation.dtype,
          bytes: activation.bytes,
        },
        values: activation.heatmap,
        flatValues: Array.isArray(activation.flat_values) ? activation.flat_values : null,
      };
      const activationAxisOrder = tensorAxisOrder(activationDatum.node.id, activationDatum.shapeInfo.shape);
      const activationValues = tensorDisplayValues(activationDatum);

      const cvs = fo.append("xhtml:canvas")
        .attr("width", d.w)
        .attr("height", d.h)
        .style("width", d.w + "px")
        .style("height", d.h + "px")
        .style("display", "block")
        .style("border-radius", "0px")
        .style("image-rendering", "pixelated");
      drawHeatmap(cvs.node(), activationValues, globalHeatRange, activation.shape, activationAxisOrder);

      const hitRect = g.append("rect")
        .attr("x", -d.w / 2)
        .attr("y", -d.h / 2)
        .attr("width", d.w)
        .attr("height", d.h)
        .style("fill", "transparent")
        .style("cursor", "pointer");

      hitRect.on("mousemove", function (event) {
        const rows = activationValues.length;
        const cols = activationValues[0]?.length || 1;
        const [px, py] = d3.pointer(event, this);
        const col = Math.max(0, Math.min(cols - 1, Math.floor((px + d.w / 2) / d.w * cols)));
        const row = Math.max(0, Math.min(rows - 1, Math.floor((py + d.h / 2) / d.h * rows)));
        const value = activationValues[row]?.[col];
        const indexLabel = tensorIndexLabel(activation.shape, row, col, activationAxisOrder);
        const rowsOut = parameterInspectorRows(activationDatum, [
          { key: "current", value: value != null ? formatHeatValue(value) : "-" },
          { key: "index", value: indexLabel },
          { key: "path", value: d.node.path || "-" },
        ]);
        showInspector("Node", rowsOut);
      });

      return;
    }

    g.append("rect")
      .attr("x", -d.w / 2)
      .attr("y", -d.h / 2)
      .attr("rx", d.node._collapsed ? 10 : 7)
      .attr("width", d.w)
      .attr("height", d.h);

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.33em")
      .text(nodeLabel(d));
  });

  /* parameter-source nodes */
  const gParam = nodeG.selectAll("g.param-node").data(paramNodesData).enter()
    .append("g").attr("class", "param-node")
    .attr("transform", (d) => `translate(${d.x},${d.y})`);

  gParam.on("mouseenter", (_, d) => {
    showInspector("Parameter", parameterInspectorRows(d));
  }).on("mouseleave", () => hideInspector());

  // Clicking a parameter node toggles its heatmap visibility (selective mode).
  // Visible heatmaps with cycling support cycle the projection on click.
  gParam.on("click", (event, d) => {
    if (event.button !== 0) return;
    if (!d.values?.length) return;
    if (paramHeatmapVisible(d.node.id)) {
      if (tensorSupportsProjectionCycling(d.shapeInfo?.shape, d.flatValues)) {
        cycleTensorProjection(d.node.id, d.shapeInfo?.shape, 1);
        renderOperationGraph(currentGraph);
      }
      return;
    }
    selectiveParamHeatmaps.add(d.node.id);
    renderOperationGraph(currentGraph);
  });

  // Right-click hides a selectively-visible parameter heatmap.
  gParam.on("contextmenu", (event, d) => {
    if (!d.values?.length) return;
    if (paramHeatmapVisible(d.node.id)) {
      if (tensorSupportsProjectionCycling(d.shapeInfo?.shape, d.flatValues) && !event.shiftKey) {
        event.preventDefault();
        cycleTensorProjection(d.node.id, d.shapeInfo?.shape, -1);
        renderOperationGraph(currentGraph);
        return;
      }
      if (selectiveParamHeatmaps.has(d.node.id)) {
        event.preventDefault();
        selectiveParamHeatmaps.delete(d.node.id);
        expandedHeatmaps.delete(d.node.id);
        renderOperationGraph(currentGraph);
      }
      return;
    }
  });

  if (showHeatmaps || selectiveParamHeatmaps.size > 0) {
    gParam.each(function (d) {
      const g = d3.select(this);
      const hasVals = d.values && d.values.length;
      const showParamHm = hasVals && paramHeatmapVisible(d.node.id);
      if (!showParamHm) {
        g.append("rect")
          .attr("x", -d.w / 2).attr("y", -d.h / 2)
          .attr("rx", 9).attr("width", d.w).attr("height", d.h);
        g.append("text").attr("text-anchor", "middle").attr("dy", "0.33em")
          .text(paramLabel(d));
        return;
      }

      // Heatmaps should always be rectangular (no rounded corners).
      const cornerR = 0;
      const clipId = `hm-clip-${d.node.id.replace(/[^a-zA-Z0-9]/g, "_")}`;
      g.append("defs").append("clipPath").attr("id", clipId)
        .append("rect").attr("x", -d.w / 2).attr("y", -d.h / 2)
        .attr("width", d.w).attr("height", d.h).attr("rx", cornerR);

      const fo = g.append("foreignObject")
        .attr("x", -d.w / 2).attr("y", -d.h / 2)
        .attr("width", d.w).attr("height", d.h)
        .attr("clip-path", `url(#${clipId})`)
        .style("pointer-events", "none");

      const displayedValues = tensorDisplayValues(d);
      const axisOrder = tensorAxisOrder(d.node.id, d.shapeInfo?.shape);
      const cvs = fo.append("xhtml:canvas")
        .attr("width", d.w).attr("height", d.h)
        .style("width", d.w + "px").style("height", d.h + "px")
        .style("display", "block")
        .style("border-radius", `${cornerR}px`)
        .style("image-rendering", "pixelated");
      drawHeatmap(cvs.node(), displayedValues, globalHeatRange, d.shapeInfo?.shape, axisOrder);

      // Invisible hit area – use .style() so CSS .param-node rect can't override
      const hitRect = g.append("rect")
        .attr("x", -d.w / 2).attr("y", -d.h / 2)
        .attr("width", d.w).attr("height", d.h)
        .style("fill", "transparent")
        .style("cursor", "pointer");

      hitRect.on("mousemove", function (event) {
        if (!displayedValues?.length) return;
        const rows = displayedValues.length;
        const cols = displayedValues[0]?.length || 1;
        const [px, py] = d3.pointer(event, this);
        const col = Math.max(0, Math.min(cols - 1, Math.floor((px + d.w / 2) / d.w * cols)));
        const row = Math.max(0, Math.min(rows - 1, Math.floor((py + d.h / 2) / d.h * rows)));
        const value = displayedValues[row]?.[col];
        const indexLabel = tensorIndexLabel(d.shapeInfo?.shape, row, col, axisOrder);
        showInspector("Parameter", parameterInspectorRows(d, [
          { key: "current", value: value != null ? formatHeatValue(value) : "-" },
          { key: "index", value: indexLabel },
        ]));
      });
    });

    updateZoomAwareOverlays(currentTransform?.k ?? 1);
  } else {
    gParam.append("rect")
      .attr("x", -paramNodeW / 2).attr("y", -paramNodeH / 2)
      .attr("rx", 9).attr("width", paramNodeW).attr("height", paramNodeH);

    gParam.append("text").attr("text-anchor", "middle").attr("dy", "0.33em")
      .text((d) => paramLabel(d));
  }

  /* INPUT / OUTPUT nodes */
  const ioData = Array.from(nodePos.values()).filter(d => d.node.type === "input" || d.node.type === "output");
  ioData.forEach(d => {
    const isInput = d.node.type === "input";
    const g = nodeG.append("g")
      .attr("class", `io-node ${isInput ? "input-node" : "output-node"}`)
      .attr("transform", `translate(${d.x},${d.y})`);

    if (isInput) {
      if (showHeatmaps && currentInputHeatmap) {
        const hmX = -d.w / 2;
        const hmY = -d.h / 2;
        const hmW = d.w;
        const hmH = d.h;

        const clipId = `input-hm-clip`;
        g.append("defs").append("clipPath").attr("id", clipId)
          .append("rect").attr("x", hmX).attr("y", hmY)
          .attr("width", hmW).attr("height", hmH).attr("rx", 0);

        const foHm = g.append("foreignObject")
          .attr("x", hmX).attr("y", hmY)
          .attr("width", hmW).attr("height", hmH)
          .attr("clip-path", `url(#${clipId})`)
          .style("pointer-events", "none");

        const inputAxisOrder = tensorAxisOrder(inputDatum.node.id, inputDatum.shapeInfo?.shape);
        const inputValues = tensorDisplayValues(inputDatum);

        const cvsHm = foHm.append("xhtml:canvas")
          .attr("width", hmW).attr("height", hmH)
          .style("width", hmW + "px").style("height", hmH + "px")
          .style("display", "block")
          .style("border-radius", "0px")
          .style("image-rendering", "pixelated");
        drawHeatmap(cvsHm.node(), inputValues, globalHeatRange, inputDatum.shapeInfo?.shape, inputAxisOrder);

        const hitRect = g.append("rect")
          .attr("x", -d.w / 2).attr("y", -d.h / 2)
          .attr("width", d.w).attr("height", d.h)
          .style("fill", "transparent")
          .style("cursor", "pointer");

        hitRect.on("mousemove", function (event) {
          const rows = inputValues.length;
          const cols = inputValues[0]?.length || 1;
          const [px, py] = d3.pointer(event, this);
          const col = Math.max(0, Math.min(cols - 1, Math.floor((px + d.w / 2) / d.w * cols)));
          const row = Math.max(0, Math.min(rows - 1, Math.floor((py + d.h / 2) / d.h * rows)));
          const value = inputValues[row]?.[col];
          const indexLabel = tensorIndexLabel(inputDatum.shapeInfo?.shape, row, col, inputAxisOrder);
          showInspector("Input", parameterInspectorRows(inputDatum, [
            { key: "current", value: value != null ? formatHeatValue(value) : "-" },
            { key: "index", value: indexLabel },
          ]));
        });

        g.on("click", function (event) {
          if (event.button !== 0) return;
          if (!tensorSupportsProjectionCycling(inputDatum.shapeInfo?.shape, inputDatum.flatValues)) return;
          cycleTensorProjection(inputDatum.node.id, inputDatum.shapeInfo?.shape, 1);
          renderOperationGraph(currentGraph);
        });

        g.on("contextmenu", function (event) {
          if (!tensorSupportsProjectionCycling(inputDatum.shapeInfo?.shape, inputDatum.flatValues)) return;
          event.preventDefault();
          cycleTensorProjection(inputDatum.node.id, inputDatum.shapeInfo?.shape, -1);
          renderOperationGraph(currentGraph);
        });

        g.on("mouseenter", () => {
          showInspector("Input", parameterInspectorRows(inputDatum));
        });
        g.on("mouseleave", () => hideInspector());
      } else {
        g.append("rect")
          .attr("x", -d.w / 2).attr("y", -d.h / 2)
          .attr("width", d.w).attr("height", d.h)
          .attr("rx", d.h / 2);

        g.append("text").attr("class", "io-label")
          .attr("text-anchor", "middle")
          .attr("y", -4)
          .text("INPUT");

        const foW = 112, foH = 18;
        const fo = g.append("foreignObject")
          .attr("x", -foW / 2).attr("y", 3)
          .attr("width", foW).attr("height", foH);
        const inp = fo.append("xhtml:input")
          .attr("type", "text")
          .attr("name", "input-shape")
          .attr("class", "graph-input-shape")
          .attr("placeholder", "1, 128")
          .attr("spellcheck", "false")
          .property("value", currentInputShapeValue);
        let shapeTimer = null;
        inp.on("input", function () {
          currentInputShapeValue = this.value;
          clearTimeout(shapeTimer);
          shapeTimer = setTimeout(() => { if (currentGraph) renderOperationGraph(currentGraph); }, 400);
        });
        inp.on("keydown", function (e) {
          if (e.key === "Enter") {
            clearTimeout(shapeTimer);
            currentInputShapeValue = this.value;
            if (currentGraph) renderOperationGraph(currentGraph);
          }
          e.stopPropagation();
        });
        inp.on("click", (e) => e.stopPropagation());

        g.on("mouseenter", () => {
          const s = parseInputShape();
          showInspector("Input", [{ key: "shape", value: s ? shapeStr(s) : "(not set)" }]);
        });
        g.on("mouseleave", () => hideInspector());
      }

      g.classed("input-drop-target", true)
        .on("dragover", (event) => {
          event.preventDefault();
          event.stopPropagation();
          g.classed("drop-ready", true);
        })
        .on("dragleave", (event) => {
          event.preventDefault();
          event.stopPropagation();
          g.classed("drop-ready", false);
        })
        .on("drop", async (event) => {
          event.preventDefault();
          event.stopPropagation();
          g.classed("drop-ready", false);
          const files = event.dataTransfer?.files;
          if (!files || !files.length) return;
          const file = files[0];
          if (!/\.npy$/i.test(file.name)) {
            setStatus("Drop a .npy tensor onto INPUT.", true);
            return;
          }
          try {
            await loadInputFromNpy(file);
          } catch (err) {
            setStatus(err?.message || "Failed to load .npy file.", true);
          }
        });
    } else {
      g.append("rect")
        .attr("x", -d.w / 2).attr("y", -d.h / 2)
        .attr("width", d.w).attr("height", d.h)
        .attr("rx", d.h / 2);

      g.append("text").attr("class", "io-label")
        .attr("text-anchor", "middle")
        .attr("y", -4)
        .text("OUTPUT");

      const outIdx = graph.edges.findIndex(e => e.target === "__OUTPUT__" && e.kind === "flow");
      const outSI = outIdx >= 0 ? edgeShapes.get(outIdx) : null;
      const outputShape = currentOutputShape || outSI?.shape || null;

      if (showHeatmaps && currentOutputHeatmap?.length) {
        const hmX = -d.w / 2;
        const hmY = -d.h / 2;
        const hmW = d.w;
        const hmH = d.h;

        const clipId = `output-hm-clip`;
        g.append("defs").append("clipPath").attr("id", clipId)
          .append("rect").attr("x", hmX).attr("y", hmY)
          .attr("width", hmW).attr("height", hmH).attr("rx", 0);

        const foHm = g.append("foreignObject")
          .attr("x", hmX).attr("y", hmY)
          .attr("width", hmW).attr("height", hmH)
          .attr("clip-path", `url(#${clipId})`)
          .style("pointer-events", "none");

        const outputRenderDatum = {
          ...outputDatum,
          shapeInfo: {
            ...(outputDatum.shapeInfo || {}),
            shape: outputShape,
            dtype: currentOutputDtype || outSI?.dtype || null,
            bytes: null,
          },
        };
        const outputAxisOrder = tensorAxisOrder(outputRenderDatum.node.id, outputRenderDatum.shapeInfo?.shape);
        const outputValues = tensorDisplayValues(outputRenderDatum);

        const cvsHm = foHm.append("xhtml:canvas")
          .attr("width", hmW).attr("height", hmH)
          .style("width", hmW + "px").style("height", hmH + "px")
          .style("display", "block")
          .style("border-radius", "0px")
          .style("image-rendering", "pixelated");
        drawHeatmap(cvsHm.node(), outputValues, globalHeatRange, outputShape, outputAxisOrder);

        const hitRect = g.append("rect")
          .attr("x", -d.w / 2).attr("y", -d.h / 2)
          .attr("width", d.w).attr("height", d.h)
          .style("fill", "transparent")
          .style("cursor", "pointer");

        hitRect.on("mousemove", function (event) {
          const rows = outputValues.length;
          const cols = outputValues[0]?.length || 1;
          const [px, py] = d3.pointer(event, this);
          const col = Math.max(0, Math.min(cols - 1, Math.floor((px + d.w / 2) / d.w * cols)));
          const row = Math.max(0, Math.min(rows - 1, Math.floor((py + d.h / 2) / d.h * rows)));
          const value = outputValues[row]?.[col];
          const indexLabel = tensorIndexLabel(outputShape, row, col, outputAxisOrder);
          const rowsOut = parameterInspectorRows(outputRenderDatum, [
            { key: "current", value: value != null ? formatHeatValue(value) : "-" },
            { key: "index", value: indexLabel },
          ]);
          if (currentOutputStats) {
            rowsOut.push(
              { key: "min", value: currentOutputStats.min.toFixed(4) },
              { key: "max", value: currentOutputStats.max.toFixed(4) },
              { key: "mean", value: currentOutputStats.mean.toFixed(4) },
              { key: "std", value: currentOutputStats.std.toFixed(4) },
            );
          }
          showInspector("Output", rowsOut);
        });

        g.on("click", function (event) {
          if (event.button !== 0) return;
          if (!tensorSupportsProjectionCycling(outputRenderDatum.shapeInfo?.shape, outputRenderDatum.flatValues)) return;
          cycleTensorProjection(outputRenderDatum.node.id, outputRenderDatum.shapeInfo?.shape, 1);
          renderOperationGraph(currentGraph);
        });

        g.on("contextmenu", function (event) {
          if (!tensorSupportsProjectionCycling(outputRenderDatum.shapeInfo?.shape, outputRenderDatum.flatValues)) return;
          event.preventDefault();
          cycleTensorProjection(outputRenderDatum.node.id, outputRenderDatum.shapeInfo?.shape, -1);
          renderOperationGraph(currentGraph);
        });

        g.on("mouseenter", () => {
          const rows = parameterInspectorRows(outputRenderDatum);
          if (currentOutputStats) {
            rows.push(
              { key: "min", value: currentOutputStats.min.toFixed(4) },
              { key: "max", value: currentOutputStats.max.toFixed(4) },
              { key: "mean", value: currentOutputStats.mean.toFixed(4) },
              { key: "std", value: currentOutputStats.std.toFixed(4) },
            );
          }
          showInspector("Output", rows);
        });
        g.on("mouseleave", () => hideInspector());
      } else {
        g.append("text").attr("class", "io-shape")
          .attr("text-anchor", "middle").attr("dy", "1em")
          .text(outputShape ? shapeStr(outputShape) : "?");

        g.on("mouseenter", () => {
          const rows = [{ key: "shape", value: outputShape ? shapeStr(outputShape) : "?" }];
          if (currentOutputDtype) rows.push({ key: "dtype", value: currentOutputDtype });
          showInspector("Output", rows);
        });
        g.on("mouseleave", () => hideInspector());
      }
    }
  });
}

/* ── Controls ───────────────────────────────────────────── */

function updateCollapseButton() {
  if (!collapseBtn) return;
  collapseBtn.textContent = allCollapsed ? "Expand All" : "Collapse All";
}

function canToggleCollapseAll() {
  if (!currentGraph) return false;
  const tops = getTopLevelPrefixes(currentGraph);
  for (const top of tops) {
    if (getChildPrefixes(currentGraph, top).size > 0) {
      return true;
    }
  }
  return false;
}

function toggleCollapseAll() {
  if (!canToggleCollapseAll()) return;
  if (allCollapsed) {
    collapsedGroups.clear();
    allCollapsed = false;
  } else {
    collapsedGroups.clear();
    const tops = getTopLevelPrefixes(currentGraph);
    tops.forEach((t) => collapsedGroups.add(t));
    allCollapsed = true;
  }
  updateCollapseButton();
  renderOperationGraph(currentGraph);
}

if (collapseBtn) {
  collapseBtn.addEventListener("click", () => {
    toggleCollapseAll();
  });
}

let lastBackgroundRightClick = 0;
const backgroundDoubleRightMs = 360;

function isGraphBackgroundTarget(target) {
  if (!(target instanceof Element)) return false;
  return !target.closest(
    ".node, .param-node, .io-node, .group-hull, .graph-input-shape"
  );
}

if (vizEl) {
  vizEl.addEventListener("dblclick", (event) => {
    if (!isGraphBackgroundTarget(event.target)) return;
    if (!canToggleCollapseAll()) return;
    event.preventDefault();
    toggleCollapseAll();
  });

  vizEl.addEventListener("contextmenu", (event) => {
    if (!isGraphBackgroundTarget(event.target)) return;
    if (!canToggleCollapseAll()) return;
    event.preventDefault();
    const now = performance.now();
    if (now - lastBackgroundRightClick <= backgroundDoubleRightMs) {
      lastBackgroundRightClick = 0;
      toggleCollapseAll();
    } else {
      lastBackgroundRightClick = now;
    }
  });
}

function updateShapesButton() {
  if (!shapesBtn) return;
  shapesBtn.textContent = "Shapes";
  shapesBtn.setAttribute("aria-pressed", showShapes ? "true" : "false");
  shapesBtn.classList.toggle("is-active", showShapes);
}

function updateDefaultButton() {
  if (!defaultBtn) return;
  const isDefault = !showShapes && !showHeatmaps;
  defaultBtn.textContent = "Default";
  defaultBtn.setAttribute("aria-pressed", isDefault ? "true" : "false");
  defaultBtn.classList.toggle("is-active", isDefault);
}

if (shapesBtn) {
  shapesBtn.addEventListener("click", () => {
    showShapes = !showShapes;
    updateDefaultButton();
    updateShapesButton();
    if (currentGraph) renderOperationGraph(currentGraph);
  });
}

if (defaultBtn) {
  defaultBtn.addEventListener("click", () => {
    showShapes = false;
    showHeatmaps = false;
    // Explicit default view clears any selectively forced parameter heatmaps.
    selectiveParamHeatmaps.clear();
    updateDefaultButton();
    updateShapesButton();
    updateHeatmapButton();
    if (currentGraph) renderOperationGraph(currentGraph);
  });
}

if (themeBtn) {
  const validThemeMode = (mode) => mode === "auto" || mode === "light" || mode === "dark";

  const resolveLightTheme = () => {
    if (themeMode === "light") return true;
    if (themeMode === "dark") return false;
    return systemLightQuery ? systemLightQuery.matches : true;
  };

  const syncThemeButton = () => {
    const isLight = resolveLightTheme();
    document.documentElement.classList.toggle("light", isLight);
    themeBtn.textContent = themeMode === "auto"
      ? "Auto"
      : themeMode === "light"
        ? "Light"
        : "Dark";
    themeBtn.classList.toggle("is-light", isLight);
    themeBtn.classList.toggle("is-auto", themeMode === "auto");
    themeBtn.setAttribute("aria-label", `Theme mode: ${themeMode}. Click to switch theme mode.`);
  };

  themeBtn.addEventListener("click", () => {
    themeMode = themeMode === "auto" ? "dark" : themeMode === "dark" ? "light" : "auto";
    try {
      localStorage.setItem("eqxview.themeMode", themeMode);
    } catch (_) {}
    syncThemeButton();
  });

  try {
    const savedThemeMode = localStorage.getItem("eqxview.themeMode");
    if (validThemeMode(savedThemeMode)) themeMode = savedThemeMode;
  } catch (_) {}

  if (systemLightQuery) {
    const onSystemThemeChange = () => {
      if (themeMode === "auto") syncThemeButton();
    };
    if (typeof systemLightQuery.addEventListener === "function") {
      systemLightQuery.addEventListener("change", onSystemThemeChange);
    } else if (typeof systemLightQuery.addListener === "function") {
      systemLightQuery.addListener(onSystemThemeChange);
    }
  }

  syncThemeButton();
}

function updateHeatmapButton() {
  if (!heatmapBtn) return;
  heatmapBtn.textContent = "Heatmaps";
  heatmapBtn.setAttribute("aria-pressed", showHeatmaps ? "true" : "false");
  heatmapBtn.classList.toggle("is-active", showHeatmaps);
}

if (heatmapBtn) {
  heatmapBtn.addEventListener("click", () => {
    showHeatmaps = !showHeatmaps;
    updateDefaultButton();
    updateHeatmapButton();
    if (currentGraph) renderOperationGraph(currentGraph);
  });
}

updateCollapseButton();
updateDefaultButton();
updateShapesButton();
updateHeatmapButton();

async function loadModelTreeFromFile(file) {
  if (!file) {
    setStatus("No file provided.", true);
    return;
  }
  setStatus(`Loading ${file.name}...`);
  setProcessing(true, `Processing ${file.name}...`);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/introspect-upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed.");
    }

    const data = await res.json();
    if (!data?.summary) {
      throw new Error("Server response missing summary.");
    }
    renderSummary(data.summary);

    const graph = data.graph && Array.isArray(data.graph.nodes)
      ? data.graph
      : graphFromTree(data.tree);
    currentGraph = graph;
    clearCurrentOutput();
    selectiveParamHeatmaps.clear();
    collapsedGroups.clear();
    // Start collapsed at top level by default.
    const tops = getTopLevelPrefixes(currentGraph);
    tops.forEach((t) => collapsedGroups.add(t));
    allCollapsed = true;
    currentTransform = null;
    updateCollapseButton();
    const runResult = await runCurrentModelWithInput();
    await renderOperationGraph(currentGraph);
    if (currentInputFile && !runResult.ok) {
      setStatus(`Loaded model, but INPUT inference failed: ${runResult.error}`, true);
    } else if (currentInputFile) {
      setStatus("Loaded and ran model on current INPUT tensor.");
    } else {
      setStatus("Loaded.");
    }
  } catch (err) {
    setStatus(err.message || "Failed to load model.", true);
  } finally {
    setProcessing(false);
  }
}

function overlayOn() {
  if (currentGraph) return;
  dropOverlayEl.classList.add("active");
}

function overlayOff() {
  dropOverlayEl.classList.remove("active");
}

function setProcessing(active, text = "Processing model...") {
  if (!processingOverlayEl) return;
  if (processingTextEl) processingTextEl.textContent = text;
  processingOverlayEl.classList.toggle("active", !!active);
  processingOverlayEl.setAttribute("aria-hidden", active ? "false" : "true");
}

window.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  overlayOn();
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    overlayOff();
  }
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  overlayOff();

  const files = event.dataTransfer?.files;
  if (!files || files.length === 0) {
    setStatus("Drop a model file to load.", true);
    return;
  }

  if (/\.npy$/i.test(files[0].name)) {
    setStatus("Drop .npy files directly onto the INPUT node.", true);
    return;
  }

  loadModelTreeFromFile(files[0]);
});
