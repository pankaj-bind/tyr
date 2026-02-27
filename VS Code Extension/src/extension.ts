/**
 * Tyr — VS Code Extension Entry Point (v0.2.0)
 *
 * Advanced features:
 *   - Rich WebView results panel with syntax highlighting
 *   - CGSC (Counterexample-Guided Self-Correction) audit trail display
 *   - Big-O complexity comparison
 *   - Status bar integration
 *   - Side-by-side diff view
 *   - One-click apply
 */

import * as vscode from "vscode";
import * as https from "https";
import * as http from "http";
import * as url from "url";

// ---------------------------------------------------------------------------
// Types matching the backend VerifyResponse (v0.2.0)
// ---------------------------------------------------------------------------

interface ComplexityInfo {
  time: string;
  space: string;
  explanation: string;
}

interface CorrectionRound {
  round: number;
  optimized_code: string;
  status: string;
  message: string;
  counterexample: Record<string, unknown> | null;
}

interface VerifyResponse {
  original_code: string;
  optimized_code: string;
  status: "UNSAT" | "SAT" | "ERROR";
  message: string;
  counterexample: Record<string, unknown> | null;
  correction_rounds: CorrectionRound[];
  total_rounds: number;
  original_complexity: ComplexityInfo | null;
  optimized_complexity: ComplexityInfo | null;
  complexity_improved: boolean | null;
  elapsed_ms: number;
}

// ---------------------------------------------------------------------------
// Virtual-file content provider for the diff view
// ---------------------------------------------------------------------------

class TyrContentProvider implements vscode.TextDocumentContentProvider {
  private contents = new Map<string, string>();
  private _onDidChange = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this._onDidChange.event;

  set(uri: vscode.Uri, content: string): void {
    this.contents.set(uri.toString(), content);
    this._onDidChange.fire(uri);
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? "";
  }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

let statusBarItem: vscode.StatusBarItem;

function createStatusBar(): vscode.StatusBarItem {
  const item = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100
  );
  item.command = "tyr.optimizeAndVerify";
  item.text = "$(shield) Tyr";
  item.tooltip = "Tyr — Click to optimize & verify selected code";
  item.show();
  return item;
}

function setStatusBarState(
  state: "idle" | "working" | "pass" | "fail" | "error"
): void {
  switch (state) {
    case "idle":
      statusBarItem.text = "$(shield) Tyr";
      statusBarItem.backgroundColor = undefined;
      statusBarItem.tooltip = "Tyr — Click to optimize & verify selected code";
      break;
    case "working":
      statusBarItem.text = "$(sync~spin) Tyr: Verifying…";
      statusBarItem.backgroundColor = undefined;
      break;
    case "pass":
      statusBarItem.text = "$(check) Tyr: UNSAT ✓";
      statusBarItem.backgroundColor = undefined;
      statusBarItem.tooltip = "Tyr — Last result: Verified Equivalent";
      break;
    case "fail":
      statusBarItem.text = "$(x) Tyr: SAT ✗";
      statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
      statusBarItem.tooltip = "Tyr — Last result: Semantics Differ";
      break;
    case "error":
      statusBarItem.text = "$(warning) Tyr: Error";
      statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground"
      );
      break;
  }
}

// ---------------------------------------------------------------------------
// WebView panel
// ---------------------------------------------------------------------------

let resultsPanel: vscode.WebviewPanel | undefined;

function getOrCreateResultsPanel(
  context: vscode.ExtensionContext
): vscode.WebviewPanel {
  if (resultsPanel) {
    resultsPanel.reveal(vscode.ViewColumn.Beside);
    return resultsPanel;
  }

  resultsPanel = vscode.window.createWebviewPanel(
    "tyrResults",
    "Tyr Results",
    vscode.ViewColumn.Beside,
    { enableScripts: true, retainContextWhenHidden: true }
  );

  resultsPanel.onDidDispose(() => {
    resultsPanel = undefined;
  });

  return resultsPanel;
}

function renderResultsPanel(
  panel: vscode.WebviewPanel,
  result: VerifyResponse
): void {
  panel.webview.html = getWebviewHtml(result);
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  const originalProvider = new TyrContentProvider();
  const optimizedProvider = new TyrContentProvider();

  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(
      "tyr-original",
      originalProvider
    ),
    vscode.workspace.registerTextDocumentContentProvider(
      "tyr-optimized",
      optimizedProvider
    )
  );

  // Status bar
  statusBarItem = createStatusBar();
  context.subscriptions.push(statusBarItem);

  // Store last result for the webview apply button
  let lastResult: VerifyResponse | undefined;
  let lastDocUri: vscode.Uri | undefined;
  let lastOriginalCode: string | undefined;

  // Handle messages from WebView
  const setupWebviewMessages = (panel: vscode.WebviewPanel) => {
    panel.webview.onDidReceiveMessage(async (msg) => {
      if (msg.command === "apply" && lastResult && lastDocUri && lastOriginalCode) {
        await applyOptimizedCode(lastDocUri, lastOriginalCode, lastResult.optimized_code);
      } else if (msg.command === "openDiff") {
        if (lastResult) {
          const timestamp = Date.now();
          const ext = languageExtension("python");
          const originalUri = vscode.Uri.parse(
            `tyr-original:Original.${ext}?t=${timestamp}`
          );
          const optimizedUri = vscode.Uri.parse(
            `tyr-optimized:Optimized.${ext}?t=${timestamp}`
          );
          originalProvider.set(originalUri, lastResult.original_code);
          optimizedProvider.set(optimizedUri, lastResult.optimized_code);
          await vscode.commands.executeCommand(
            "vscode.diff",
            originalUri,
            optimizedUri,
            `Tyr: Original ↔ Optimized [${lastResult.status}]`
          );
        }
      }
    });
  };

  const disposable = vscode.commands.registerCommand(
    "tyr.optimizeAndVerify",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("Tyr: No active editor found.");
        return;
      }

      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);

      if (!selectedText || selectedText.trim().length === 0) {
        vscode.window.showWarningMessage(
          "Tyr: Please select a block of code first."
        );
        return;
      }

      lastDocUri = editor.document.uri;
      lastOriginalCode = selectedText;

      const languageId = editor.document.languageId ?? "python";
      const config = vscode.workspace.getConfiguration("tyr");
      const backendUrl = config.get<string>("backendUrl", "http://localhost:8000");
      const endpoint = `${backendUrl}/verify`;

      setStatusBarState("working");

      let result: VerifyResponse;

      try {
        result = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: "Tyr",
            cancellable: false,
          },
          async (progress) => {
            progress.report({ message: "Sending to LLM for optimization…" });

            const resp = await postJSON<VerifyResponse>(endpoint, {
              code: selectedText,
              language: languageId,
            });

            return resp;
          }
        );
      } catch (err: unknown) {
        setStatusBarState("error");
        const msg = err instanceof Error ? err.message : String(err);
        vscode.window.showErrorMessage(`Tyr: Backend request failed — ${msg}`);
        return;
      }

      lastResult = result;

      // Update status bar
      if (result.status === "UNSAT") {
        setStatusBarState("pass");
      } else if (result.status === "SAT") {
        setStatusBarState("fail");
      } else {
        setStatusBarState("error");
      }

      // Open WebView results panel
      const panel = getOrCreateResultsPanel(context);
      setupWebviewMessages(panel);
      renderResultsPanel(panel, result);

      // Also open diff view for convenience
      const timestamp = Date.now();
      const ext = languageExtension(languageId);
      const originalUri = vscode.Uri.parse(
        `tyr-original:Original.${ext}?t=${timestamp}`
      );
      const optimizedUri = vscode.Uri.parse(
        `tyr-optimized:Optimized.${ext}?t=${timestamp}`
      );

      originalProvider.set(originalUri, result.original_code);
      optimizedProvider.set(optimizedUri, result.optimized_code);

      await vscode.commands.executeCommand(
        "vscode.diff",
        originalUri,
        optimizedUri,
        `Tyr: Original ↔ Optimized  [${result.status}]`
      );
    }
  );

  context.subscriptions.push(disposable);
  console.log('Extension "Tyr" v0.2.0 is now active.');
}

export function deactivate(): void {
  statusBarItem?.dispose();
}

// ---------------------------------------------------------------------------
// HTTP helper — zero-dependency POST using Node built-ins
// ---------------------------------------------------------------------------

function postJSON<T>(
  endpoint: string,
  body: Record<string, unknown>
): Promise<T> {
  return new Promise((resolve, reject) => {
    const parsed = new url.URL(endpoint);
    const payload = Buffer.from(JSON.stringify(body), "utf-8");

    const options: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
      path: parsed.pathname + parsed.search,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": payload.length,
      },
      timeout: 180_000, // 3 min — accounts for CGSC retries
    };

    const transport = parsed.protocol === "https:" ? https : http;

    const req = transport.request(options, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => chunks.push(chunk));
      res.on("end", () => {
        const raw = Buffer.concat(chunks).toString("utf-8");
        if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`HTTP ${res.statusCode}: ${raw}`));
          return;
        }
        try {
          resolve(JSON.parse(raw) as T);
        } catch {
          reject(new Error(`Invalid JSON response: ${raw.slice(0, 300)}`));
        }
      });
    });

    req.on("error", (err) => reject(err));
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out (180 s)"));
    });

    req.write(payload);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Apply optimized code back into the editor
// ---------------------------------------------------------------------------

async function applyOptimizedCode(
  docUri: vscode.Uri,
  originalCode: string,
  optimizedCode: string
): Promise<void> {
  // Find the editor for this document, or open it
  let editor = vscode.window.visibleTextEditors.find(
    (e) => e.document.uri.toString() === docUri.toString()
  );
  if (!editor) {
    const doc = await vscode.workspace.openTextDocument(docUri);
    editor = await vscode.window.showTextDocument(doc);
  }

  // Find the original code in the document by text search
  const docText = editor.document.getText();
  const idx = docText.indexOf(originalCode);

  if (idx === -1) {
    vscode.window.showErrorMessage(
      "Tyr: Could not find original code in the document. It may have been modified."
    );
    return;
  }

  const startPos = editor.document.positionAt(idx);
  const endPos = editor.document.positionAt(idx + originalCode.length);
  const range = new vscode.Range(startPos, endPos);

  const success = await editor.edit((editBuilder) => {
    editBuilder.replace(range, optimizedCode);
  });

  if (success) {
    vscode.window.showInformationMessage(
      "Tyr: Optimized code applied to the editor."
    );
  } else {
    vscode.window.showErrorMessage(
      "Tyr: Failed to apply optimized code."
    );
  }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function languageExtension(languageId: string): string {
  const map: Record<string, string> = {
    python: "py",
    javascript: "js",
    typescript: "ts",
    java: "java",
    c: "c",
    cpp: "cpp",
    csharp: "cs",
    go: "go",
    rust: "rs",
  };
  return map[languageId] ?? "txt";
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// WebView HTML — Rich Results Panel
// ---------------------------------------------------------------------------

function getWebviewHtml(r: VerifyResponse): string {
  const statusBadge =
    r.status === "UNSAT"
      ? '<span class="badge badge-pass">UNSAT — Verified Equivalent ✓</span>'
      : r.status === "SAT"
      ? '<span class="badge badge-fail">SAT — Semantics Differ ✗</span>'
      : '<span class="badge badge-error">ERROR — Verification Incomplete</span>';

  const origCx = r.original_complexity;
  const optCx = r.optimized_complexity;

  // Complexity improvement indicator
  const cxImproved = r.complexity_improved;
  const cxBadge =
    cxImproved === true
      ? '<div class="cx-badge cx-improved">⬆ Complexity Improved</div>'
      : cxImproved === false
      ? '<div class="cx-badge cx-same">⚠ No Complexity Improvement — Same Big-O</div>'
      : '';

  const complexityHtml =
    origCx && optCx
      ? `
    <div class="section">
      <h2>Big-O Complexity Analysis</h2>
      ${cxBadge}
      <div class="complexity-grid">
        <div class="complexity-card">
          <div class="complexity-label">Original</div>
          <div class="complexity-value">${escapeHtml(origCx.time)}</div>
          <div class="complexity-space">Space: ${escapeHtml(origCx.space)}</div>
          <div class="complexity-explanation">${escapeHtml(origCx.explanation)}</div>
        </div>
        <div class="complexity-arrow">${cxImproved === true ? '⟶' : '='}</div>
        <div class="complexity-card ${cxImproved === true ? 'optimized' : cxImproved === false ? 'no-improvement' : 'optimized'}">
          <div class="complexity-label">Optimized</div>
          <div class="complexity-value">${escapeHtml(optCx.time)}</div>
          <div class="complexity-space">Space: ${escapeHtml(optCx.space)}</div>
          <div class="complexity-explanation">${escapeHtml(optCx.explanation)}</div>
        </div>
      </div>
    </div>`
      : "";

  const counterexampleHtml = r.counterexample
    ? `
    <div class="section">
      <h2>Counterexample</h2>
      <pre class="counterexample">${escapeHtml(
        JSON.stringify(r.counterexample, null, 2)
      )}</pre>
    </div>`
    : "";

  // CGSC audit trail
  let auditHtml = "";
  if (r.correction_rounds && r.correction_rounds.length > 1) {
    const rows = r.correction_rounds
      .map((cr) => {
        const statusIcon =
          cr.status === "UNSAT" ? "✓" : cr.status === "SAT" ? "✗" : "⚠";
        const statusClass =
          cr.status === "UNSAT"
            ? "round-pass"
            : cr.status === "SAT"
            ? "round-fail"
            : "round-error";
        const ce = cr.counterexample
          ? escapeHtml(JSON.stringify(cr.counterexample))
          : "—";
        return `
          <tr class="${statusClass}">
            <td>${cr.round}</td>
            <td>${statusIcon} ${cr.status}</td>
            <td>${escapeHtml(cr.message.slice(0, 100))}</td>
            <td class="ce-cell">${ce}</td>
          </tr>`;
      })
      .join("");

    auditHtml = `
    <div class="section">
      <h2>CGSC Audit Trail — ${r.correction_rounds.length} Round${
      r.correction_rounds.length > 1 ? "s" : ""
    }</h2>
      <p class="description">Counterexample-Guided Self-Correction: each round feeds the counterexample back to the LLM for repair.</p>
      <table class="audit-table">
        <thead><tr><th>Round</th><th>Status</th><th>Message</th><th>Counterexample</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }

  const applyBtn =
    r.status === "UNSAT"
      ? '<button class="btn btn-apply" onclick="apply()">Apply Optimized Code</button>'
      : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Tyr Results</title>
<style>
  :root {
    --bg: #1e1e1e;
    --surface: #252526;
    --surface2: #2d2d30;
    --border: #3e3e42;
    --text: #cccccc;
    --text-dim: #969696;
    --accent: #569cd6;
    --green: #4ec9b0;
    --red: #f44747;
    --orange: #ce9178;
    --yellow: #dcdcaa;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.6;
  }

  .header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }

  .header h1 {
    font-size: 24px;
    font-weight: 600;
    color: var(--accent);
  }

  .header .meta {
    font-size: 13px;
    color: var(--text-dim);
  }

  .badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.5px;
  }

  .badge-pass {
    background: rgba(78, 201, 176, 0.15);
    color: var(--green);
    border: 1px solid rgba(78, 201, 176, 0.3);
  }

  .badge-fail {
    background: rgba(244, 71, 71, 0.15);
    color: var(--red);
    border: 1px solid rgba(244, 71, 71, 0.3);
  }

  .badge-error {
    background: rgba(206, 145, 120, 0.15);
    color: var(--orange);
    border: 1px solid rgba(206, 145, 120, 0.3);
  }

  .section {
    margin-bottom: 24px;
    background: var(--surface);
    border-radius: 8px;
    padding: 20px;
    border: 1px solid var(--border);
  }

  .section h2 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--accent);
  }

  .description {
    font-size: 13px;
    color: var(--text-dim);
    margin-bottom: 12px;
  }

  .complexity-grid {
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
  }

  .complexity-card {
    flex: 1;
    min-width: 180px;
    background: var(--surface2);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid var(--border);
  }

  .complexity-card.optimized {
    border-color: var(--green);
  }

  .complexity-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }

  .complexity-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--yellow);
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  }

  .complexity-space {
    font-size: 14px;
    color: var(--text-dim);
    margin-top: 4px;
  }

  .complexity-explanation {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 8px;
    font-style: italic;
  }

  .complexity-arrow {
    font-size: 32px;
    color: var(--green);
    font-weight: bold;
  }

  .complexity-card.no-improvement {
    border-color: var(--orange);
  }

  .complexity-card.no-improvement .complexity-value {
    color: var(--orange);
  }

  .cx-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 14px;
  }

  .cx-improved {
    background: rgba(78, 201, 176, 0.12);
    color: var(--green);
    border: 1px solid rgba(78, 201, 176, 0.3);
  }

  .cx-same {
    background: rgba(206, 145, 120, 0.12);
    color: var(--orange);
    border: 1px solid rgba(206, 145, 120, 0.3);
  }

  pre {
    background: var(--surface2);
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.5;
    border: 1px solid var(--border);
  }

  .counterexample {
    color: var(--orange);
  }

  .code-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .code-block h3 {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }

  .audit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  .audit-table th {
    text-align: left;
    padding: 8px 12px;
    background: var(--surface2);
    border-bottom: 2px solid var(--border);
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 11px;
  }

  .audit-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }

  .ce-cell {
    max-width: 250px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: monospace;
    font-size: 12px;
  }

  .round-pass td:nth-child(2) { color: var(--green); }
  .round-fail td:nth-child(2) { color: var(--red); }
  .round-error td:nth-child(2) { color: var(--orange); }

  .btn {
    display: inline-block;
    padding: 10px 24px;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-apply {
    background: var(--green);
    color: #1e1e1e;
    margin-right: 8px;
  }

  .btn-apply:hover {
    background: #5fd9c0;
  }

  .btn-diff {
    background: var(--accent);
    color: #1e1e1e;
  }

  .btn-diff:hover {
    background: #6aadea;
  }

  .actions {
    margin-top: 20px;
    display: flex;
    gap: 12px;
  }

  .message-box {
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 13px;
    margin-top: 12px;
  }

  .message-pass {
    background: rgba(78, 201, 176, 0.1);
    border-left: 3px solid var(--green);
    color: var(--green);
  }

  .message-fail {
    background: rgba(244, 71, 71, 0.1);
    border-left: 3px solid var(--red);
    color: var(--red);
  }

  .message-error {
    background: rgba(206, 145, 120, 0.1);
    border-left: 3px solid var(--orange);
    color: var(--orange);
  }

  @media (max-width: 600px) {
    .code-grid { grid-template-columns: 1fr; }
    .complexity-grid { flex-direction: column; }
    .complexity-arrow { transform: rotate(90deg); }
  }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ Tyr</h1>
  <div>
    ${statusBadge}
    <div class="meta">${r.total_rounds} verification round${
    r.total_rounds !== 1 ? "s" : ""
  } · ${r.elapsed_ms}ms total</div>
  </div>
</div>

<div class="section">
  <h2>Verification Result</h2>
  <div class="message-box ${
    r.status === "UNSAT"
      ? "message-pass"
      : r.status === "SAT"
      ? "message-fail"
      : "message-error"
  }">
    ${escapeHtml(r.message)}
  </div>
  <div class="actions">
    ${applyBtn}
    <button class="btn btn-diff" onclick="openDiff()">Open Diff View</button>
  </div>
</div>

${complexityHtml}

<div class="section">
  <h2>Code Comparison</h2>
  <div class="code-grid">
    <div class="code-block">
      <h3>Original</h3>
      <pre>${escapeHtml(r.original_code)}</pre>
    </div>
    <div class="code-block">
      <h3>Optimized</h3>
      <pre>${escapeHtml(r.optimized_code)}</pre>
    </div>
  </div>
</div>

${counterexampleHtml}

${auditHtml}

<script>
  const vscode = acquireVsCodeApi();

  function apply() {
    vscode.postMessage({ command: 'apply' });
  }

  function openDiff() {
    vscode.postMessage({ command: 'openDiff' });
  }
</script>

</body>
</html>`;
}
