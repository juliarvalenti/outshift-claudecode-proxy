#!/usr/bin/env node

const http = require("http");
const https = require("https");
const url = require("url");
const fs = require("fs");
const path = require("path");

const TARGET_HOST = "litellm.prod.outshift.ai";
const TARGET_PORT = 443;
const PROXY_PORT = 8099;

const DEFAULT_MODEL = "bedrock/global.anthropic.claude-sonnet-4-6";

const MODEL_MAP = {
  "claude-haiku-4-5-20251001":
    "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
  "claude-sonnet-4-5": DEFAULT_MODEL,
  "claude-sonnet-4": DEFAULT_MODEL,
  "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0": DEFAULT_MODEL,
};

// Content block types that LiteLLM / Bedrock actually understand.
// Everything else (thinking, redacted_thinking, server_tool_use,
// web_search_tool_result, citations, etc.) gets stripped so the
// proxy stays forward-compatible as Claude Code adds new beta features.
const ALLOWED_CONTENT_BLOCK_TYPES = new Set([
  "text",
  "image",
  "tool_use",
  "tool_result",
]);

const VERBOSE =
  process.argv.includes("--verbose") || process.argv.includes("-v");
const SUMMARY =
  process.argv.includes("--summary") || process.argv.includes("-s");
const LOG_REQUESTS =
  process.argv.includes("--log-requests") || process.argv.includes("-r");
const LOG_DIR = "request-logs";
const MAX_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 200;
const RETRYABLE_ERRORS = new Set(["ECONNRESET", "ETIMEDOUT", "EPIPE"]);
let requestCounter = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const safeParseJson = (value) => {
  if (typeof value !== "string" || value.length === 0) return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

const ensureLogDir = () => {
  if (!LOG_REQUESTS) return;
  fs.mkdirSync(LOG_DIR, { recursive: true });
};

const writeJsonlLog = (entry, requestId) => {
  if (!LOG_REQUESTS) return;
  ensureLogDir();
  const filePath = path.join(LOG_DIR, `${requestId}.jsonl`);
  fs.appendFileSync(filePath, `${JSON.stringify(entry)}\n`);
};

const log = (message, ...args) => {
  if (VERBOSE) console.log(message, ...args);
};

const info = (message, ...args) => console.log(message, ...args);

const summary = (message, ...args) => {
  if (SUMMARY && !VERBOSE) console.log(message, ...args);
};

const isRetryableError = (err) =>
  err && typeof err.code === "string" && RETRYABLE_ERRORS.has(err.code);

// ---------------------------------------------------------------------------
// Request body sanitisation — each function returns true if it changed anything
// ---------------------------------------------------------------------------

function fixThinkingBudget(body) {
  if (!body.thinking?.budget_tokens) return false;
  const maxTokens = body.max_tokens || 32000;
  if (body.thinking.budget_tokens < maxTokens) return false;

  const old = body.thinking.budget_tokens;
  body.thinking.budget_tokens = Math.floor(maxTokens * 0.75);
  info(`🧠 Fixed thinking budget: ${old} → ${body.thinking.budget_tokens}`);
  return true;
}

function removeUnsupportedTopLevelFields(body) {
  let changed = false;
  if (body.context_management) {
    delete body.context_management;
    info("🧹 Removed context_management");
    changed = true;
  }
  return changed;
}

function removeSystemCacheControl(body) {
  if (!Array.isArray(body.system)) return false;
  let changed = false;
  for (const entry of body.system) {
    if (entry?.cache_control) {
      delete entry.cache_control;
      changed = true;
    }
  }
  if (changed) info("🧹 Removed system cache_control fields");
  return changed;
}

function removeToolDeferLoading(body) {
  if (!Array.isArray(body.tools)) return false;
  let changed = false;
  for (const tool of body.tools) {
    if (tool && "defer_loading" in tool) {
      delete tool.defer_loading;
      changed = true;
    }
    if (tool?.custom && "defer_loading" in tool.custom) {
      delete tool.custom.defer_loading;
      changed = true;
    }
  }
  if (changed) info("🧹 Removed tools defer_loading fields");
  return changed;
}

/**
 * Strip every content block whose type is not in the whitelist.
 * This catches thinking, redacted_thinking, server_tool_use,
 * web_search_tool_result, citations, and any future beta types
 * that LiteLLM / Bedrock would reject.
 *
 * Also strips nested tool_reference entries inside tool_result blocks
 * and removes cache_control from individual content blocks.
 */
function stripNonStandardContentBlocks(body) {
  if (!Array.isArray(body.messages)) return false;

  let changed = false;
  const cleaned = [];

  for (const message of body.messages) {
    if (!Array.isArray(message?.content)) {
      cleaned.push(message);
      continue;
    }

    const filteredContent = [];
    for (const block of message.content) {
      if (!block) continue;

      // Drop blocks LiteLLM doesn't understand
      if (!ALLOWED_CONTENT_BLOCK_TYPES.has(block.type)) {
        changed = true;
        continue;
      }

      // Inside tool_result blocks, strip nested non-standard types
      if (block.type === "tool_result" && Array.isArray(block.content)) {
        const before = block.content.length;
        block.content = block.content.filter(
          (item) => item && ALLOWED_CONTENT_BLOCK_TYPES.has(item.type),
        );
        if (block.content.length !== before) changed = true;
        if (block.content.length === 0) {
          // tool_result with no content left — keep it but with empty string
          block.content = "";
        }
      }

      // Strip cache_control on individual content blocks
      if (block.cache_control) {
        delete block.cache_control;
        changed = true;
      }

      filteredContent.push(block);
    }

    // Drop messages that became empty after filtering
    if (filteredContent.length === 0) {
      changed = true;
      continue;
    }

    message.content = filteredContent;
    cleaned.push(message);
  }

  if (changed) {
    body.messages = cleaned;
    info("🧹 Stripped non-standard content blocks");
  }
  return changed;
}

/**
 * Ensure every tool_use in an assistant message has a matching tool_result
 * in the next user message, and vice-versa.  Orphans are removed.
 */
function sanitizeToolPairs(body) {
  if (!Array.isArray(body.messages)) return false;

  let changed = false;
  const out = [];

  for (let i = 0; i < body.messages.length; i++) {
    const msg = body.messages[i];

    // --- assistant: drop tool_use blocks with no matching tool_result ---
    if (msg?.role === "assistant" && Array.isArray(msg.content)) {
      const toolIds = msg.content
        .filter((b) => b?.type === "tool_use" && b.id)
        .map((b) => b.id);

      if (toolIds.length > 0) {
        const next = body.messages[i + 1];
        const hasMatch =
          next?.role === "user" &&
          Array.isArray(next.content) &&
          next.content.some(
            (b) => b?.type === "tool_result" && toolIds.includes(b.tool_use_id),
          );
        if (!hasMatch) {
          msg.content = msg.content.filter((b) => b?.type !== "tool_use");
          changed = true;
        }
      }

      if (msg.content.length === 0) {
        changed = true;
        continue; // drop empty assistant message
      }
      out.push(msg);
      continue;
    }

    // --- user: drop tool_result blocks with no matching tool_use ---
    if (msg?.role === "user" && Array.isArray(msg.content)) {
      const prev = body.messages[i - 1];
      const validIds = new Set(
        prev?.role === "assistant" && Array.isArray(prev.content)
          ? prev.content
              .filter((b) => b?.type === "tool_use" && b.id)
              .map((b) => b.id)
          : [],
      );

      const before = msg.content.length;
      msg.content = msg.content.filter(
        (b) =>
          b?.type !== "tool_result" ||
          (b.tool_use_id && validIds.has(b.tool_use_id)),
      );
      if (msg.content.length !== before) changed = true;

      if (msg.content.length === 0) {
        changed = true;
        continue;
      }
      out.push(msg);
      continue;
    }

    out.push(msg);
  }

  if (changed) {
    body.messages = out;
    info("🧹 Sanitized tool_use/tool_result pairs");
  }
  return changed;
}

function rewriteModel(body) {
  if (!body.model) return false;
  const original = body.model;

  for (const [pattern, bedrockModel] of Object.entries(MODEL_MAP)) {
    if (body.model === pattern || body.model.includes(pattern)) {
      body.model = bedrockModel;
      if (pattern.includes("haiku")) {
        info("🔀 Intercepting haiku → bedrock haiku");
      } else if (pattern.includes("sonnet")) {
        info("🔀 Intercepting sonnet → bedrock sonnet");
      }
      log(`Rewriting model: "${original}" → "${bedrockModel}"`);
      return true;
    }
  }

  info(`ℹ️  Model passthrough: ${original}`);
  return false;
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

const server = http.createServer((req, res) => {
  log(`\n${new Date().toISOString()} ${req.method} ${req.url}`);

  const parsedUrl = url.parse(req.url, true);
  delete parsedUrl.query.beta;
  delete parsedUrl.search;
  const cleanPath = url.format(parsedUrl);

  let requestId = null;
  let responseStarted = false;
  let attempt = 0;

  const chunks = [];
  req.on("data", (chunk) => chunks.push(chunk));

  req.on("end", () => {
    const rawBody = Buffer.concat(chunks);
    let modifiedBody = rawBody;
    let bodyModified = false;
    let parsedBodyJson = null;

    if (
      rawBody.length > 0 &&
      req.headers["content-type"]?.includes("application/json")
    ) {
      try {
        parsedBodyJson = JSON.parse(rawBody.toString());
        const body = parsedBodyJson;

        // Verbose / summary logging
        if (VERBOSE) {
          log("Request body:", JSON.stringify(body, null, 2));
        } else if (SUMMARY && Array.isArray(body.messages)) {
          const last = body.messages[body.messages.length - 1];
          let preview = "";
          if (typeof last?.content === "string") {
            preview = last.content;
          } else if (Array.isArray(last?.content)) {
            const t = last.content.find(
              (b) => b?.type === "text" && typeof b.text === "string",
            );
            preview = t?.text || "";
          }
          summary(
            `📝 Messages: ${body.messages.length}, last: "${preview.slice(0, 50)}"`,
          );
        }

        // Run all sanitisations (order matters for tool pairs)
        const changed =
          fixThinkingBudget(body) |
          removeUnsupportedTopLevelFields(body) |
          removeSystemCacheControl(body) |
          removeToolDeferLoading(body) |
          stripNonStandardContentBlocks(body) |
          sanitizeToolPairs(body) |
          rewriteModel(body);

        if (changed) {
          modifiedBody = Buffer.from(JSON.stringify(body));
          bodyModified = true;
        }
      } catch (e) {
        log("Failed to parse body as JSON:", e.message);
      }
    }

    // Request logging
    if (LOG_REQUESTS) {
      const timestamp = new Date().toISOString();
      requestCounter += 1;
      requestId = `${timestamp.replace(/[:.]/g, "-")}-${process.pid}-${requestCounter}`;
      writeJsonlLog(
        {
          type: "request",
          requestId,
          timestamp,
          method: req.method,
          url: req.url,
          cleanPath,
          headers: req.headers,
          originalBody: parsedBodyJson ?? rawBody.toString(),
          forwardedBody: safeParseJson(modifiedBody.toString()),
        },
        requestId,
      );
    }

    // Prepare forwarding headers
    const forwardHeaders = { ...req.headers };
    delete forwardHeaders["anthropic-beta"];
    forwardHeaders.host = TARGET_HOST;
    if (bodyModified) {
      forwardHeaders["content-length"] = modifiedBody.length;
    }

    log("Forwarding headers:", JSON.stringify(forwardHeaders, null, 2));
    log("Clean path:", cleanPath);

    const options = {
      hostname: TARGET_HOST,
      port: TARGET_PORT,
      path: cleanPath,
      method: req.method,
      headers: forwardHeaders,
    };

    const sendProxyRequest = () => {
      attempt += 1;

      const proxyReq = https.request(options, (proxyRes) => {
        log(`Response status: ${proxyRes.statusCode}`);
        responseStarted = true;
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);

        if (LOG_REQUESTS) {
          const respChunks = [];
          proxyRes.on("data", (c) => respChunks.push(c));
          proxyRes.on("end", () => {
            writeJsonlLog(
              {
                type: "response",
                requestId,
                timestamp: new Date().toISOString(),
                statusCode: proxyRes.statusCode,
                headers: proxyRes.headers,
                body: safeParseJson(Buffer.concat(respChunks).toString()),
              },
              requestId,
            );
          });
        }

        if (VERBOSE) {
          const respChunks = [];
          proxyRes.on("data", (c) => respChunks.push(c));
          proxyRes.on("end", () => {
            if (proxyRes.statusCode >= 400) {
              log("Error response:", Buffer.concat(respChunks).toString());
            }
          });
        }
      });

      proxyReq.on("error", (err) => {
        if (!responseStarted && attempt <= MAX_RETRIES && isRetryableError(err)) {
          const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1);
          log(`Retrying in ${delay}ms (${err.code}, attempt ${attempt}/${MAX_RETRIES})`);
          setTimeout(sendProxyRequest, delay);
          return;
        }

        console.error("Proxy request error:", err);
        if (!res.headersSent) {
          res.writeHead(502, { "Content-Type": "application/json" });
        }
        if (!res.writableEnded) {
          res.end(JSON.stringify({ error: "Proxy error", details: err.message }));
        }
      });

      if (modifiedBody.length > 0) proxyReq.write(modifiedBody);
      proxyReq.end();
    };

    sendProxyRequest();
  });
});

server.listen(PROXY_PORT, () => {
  console.log(`Proxy listening on http://localhost:${PROXY_PORT}`);
  console.log(`Forwarding to https://${TARGET_HOST}:${TARGET_PORT}`);
  console.log(`Stripping anthropic-beta headers and ?beta query params`);
  if (LOG_REQUESTS) console.log(`Request logging: ${LOG_DIR}/`);
  console.log(`Model mappings:`);
  for (const [from, to] of Object.entries(MODEL_MAP)) {
    console.log(`  ${from} → ${to}`);
  }
  console.log(
    `Allowed content block types: ${[...ALLOWED_CONTENT_BLOCK_TYPES].join(", ")}`,
  );
  if (VERBOSE) console.log(`Verbose mode enabled`);
  else if (SUMMARY) console.log(`Summary mode enabled`);
  else console.log(`Run with --verbose or -v for detailed logs`);
  console.log();
});
