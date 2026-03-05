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

// Model mappings
const MODEL_MAP = {
  "claude-haiku-4-5-20251001":
    "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
  "claude-sonnet-4-5": DEFAULT_MODEL,
  "claude-sonnet-4": DEFAULT_MODEL,
  "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0": DEFAULT_MODEL,
};

// Parse command line args
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

const safeParseJson = (value) => {
  if (typeof value !== "string" || value.length === 0) {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

const ensureLogDir = () => {
  if (!LOG_REQUESTS) {
    return;
  }
  fs.mkdirSync(LOG_DIR, { recursive: true });
};

const writeJsonlLog = (entry, requestId) => {
  if (!LOG_REQUESTS) {
    return;
  }
  ensureLogDir();
  const filePath = path.join(LOG_DIR, `${requestId}.jsonl`);
  fs.appendFileSync(filePath, `${JSON.stringify(entry)}\n`);
};

const log = (message, ...args) => {
  if (VERBOSE) {
    console.log(message, ...args);
  }
};

const info = (message, ...args) => {
  console.log(message, ...args);
};

const summary = (message, ...args) => {
  if (SUMMARY && !VERBOSE) {
    console.log(message, ...args);
  }
};

const isRetryableError = (err) =>
  err && typeof err.code === "string" && RETRYABLE_ERRORS.has(err.code);

const server = http.createServer((req, res) => {
  if (VERBOSE) {
    log(`\n${new Date().toISOString()} ${req.method} ${req.url}`);
  }

  // Parse the URL to strip beta query parameter
  const parsedUrl = url.parse(req.url, true);
  delete parsedUrl.query.beta;
  delete parsedUrl.search; // Clear search to force regeneration from query
  const cleanPath = url.format(parsedUrl);

  let requestId = null;
  let requestLogEntry = null;
  let responseStarted = false;
  let attempt = 0;

  // Collect request body
  let body = [];
  req.on("data", (chunk) => {
    body.push(chunk);
  });

  req.on("end", () => {
    body = Buffer.concat(body);
    const bodyStr = body.length > 0 ? body.toString() : "";
    let parsedBodyJson = null;

    // Parse and modify body if it contains a model field
    let modifiedBody = body;
    let modelRewritten = false;
    let originalModel = null;
    let newModel = null;

    if (
      body.length > 0 &&
      req.headers["content-type"]?.includes("application/json")
    ) {
      try {
        parsedBodyJson = JSON.parse(bodyStr);
        const bodyJson = parsedBodyJson;

        // Log body in verbose mode
        if (VERBOSE) {
          log("Request body:", JSON.stringify(bodyJson, null, 2));
        } else if (SUMMARY && Array.isArray(bodyJson.messages)) {
          const messagesCount = bodyJson.messages.length;
          const lastMessage = bodyJson.messages[messagesCount - 1];
          let lastText = "";
          if (lastMessage) {
            if (typeof lastMessage.content === "string") {
              lastText = lastMessage.content;
            } else if (Array.isArray(lastMessage.content)) {
              const textPart = lastMessage.content.find(
                (item) =>
                  item && item.type === "text" && typeof item.text === "string",
              );
              lastText = textPart?.text || "";
            }
          }
          const preview = lastText.slice(0, 50);
          summary(`📝 Messages: ${messagesCount}, last preview: "${preview}"`);
        }

        // Fix thinking.budget_tokens if it's >= max_tokens
        if (bodyJson.thinking && bodyJson.thinking.budget_tokens) {
          const maxTokens = bodyJson.max_tokens || 32000;
          if (bodyJson.thinking.budget_tokens >= maxTokens) {
            const oldBudget = bodyJson.thinking.budget_tokens;
            // Set budget to 75% of max_tokens to leave room for response
            bodyJson.thinking.budget_tokens = Math.floor(maxTokens * 0.75);
            if (VERBOSE) {
              log(
                `Fixed thinking.budget_tokens: ${oldBudget} → ${bodyJson.thinking.budget_tokens} (max_tokens: ${maxTokens})`,
              );
            } else {
              info(
                `🧠 Fixed thinking budget: ${oldBudget} → ${bodyJson.thinking.budget_tokens}`,
              );
            }
            modifiedBody = Buffer.from(JSON.stringify(bodyJson));
            modelRewritten = true;
          }
        }

        // Remove unsupported fields
        if (bodyJson.context_management) {
          if (VERBOSE) {
            log("Removing unsupported context_management field");
          } else {
            info("🧹 Removing context_management field");
          }
          delete bodyJson.context_management;
          modifiedBody = Buffer.from(JSON.stringify(bodyJson));
          modelRewritten = true;
        }

        // Remove unsupported cache_control fields from system messages
        if (Array.isArray(bodyJson.system)) {
          let removedCacheControl = false;
          for (const entry of bodyJson.system) {
            if (entry && entry.cache_control) {
              delete entry.cache_control;
              removedCacheControl = true;
            }
          }
          if (removedCacheControl) {
            if (VERBOSE) {
              log("Removing unsupported system.cache_control fields");
            } else {
              info("🧹 Removing system cache_control fields");
            }
            modifiedBody = Buffer.from(JSON.stringify(bodyJson));
            modelRewritten = true;
          }
        }

        // Remove unsupported defer_loading field from tools payloads
        if (Array.isArray(bodyJson.tools)) {
          let removedDeferLoading = false;
          for (const tool of bodyJson.tools) {
            if (tool && "defer_loading" in tool) {
              delete tool.defer_loading;
              removedDeferLoading = true;
            }
            if (tool && tool.custom && "defer_loading" in tool.custom) {
              delete tool.custom.defer_loading;
              removedDeferLoading = true;
            }
          }
          if (removedDeferLoading) {
            if (VERBOSE) {
              log("Removing unsupported tools defer_loading fields");
            } else {
              info("🧹 Removing tools defer_loading fields");
            }
            modifiedBody = Buffer.from(JSON.stringify(bodyJson));
            modelRewritten = true;
          }
        }

        // Remove unsupported tool_reference entries from tool_result content
        if (Array.isArray(bodyJson.messages)) {
          let removedToolReference = false;
          for (const message of bodyJson.messages) {
            if (!Array.isArray(message?.content)) {
              continue;
            }
            const filteredContent = [];
            for (const item of message.content) {
              if (
                item &&
                item.type === "tool_result" &&
                Array.isArray(item.content)
              ) {
                const filteredToolContent = item.content.filter(
                  (toolItem) => toolItem && toolItem.type !== "tool_reference",
                );
                if (filteredToolContent.length !== item.content.length) {
                  removedToolReference = true;
                  if (filteredToolContent.length > 0) {
                    item.content = filteredToolContent;
                  } else {
                    continue;
                  }
                }
              }
              filteredContent.push(item);
            }
            if (filteredContent.length !== message.content.length) {
              message.content = filteredContent;
            }
          }
          if (removedToolReference) {
            if (VERBOSE) {
              log(
                "Removing unsupported tool_reference entries from tool_result content",
              );
            } else {
              info(
                "🧹 Removing tool_reference entries from tool_result content",
              );
            }
            modifiedBody = Buffer.from(JSON.stringify(bodyJson));
            modelRewritten = true;
          }
        }

        // Remove assistant thinking blocks
        if (Array.isArray(bodyJson.messages)) {
          let removedThinking = false;
          const filteredMessages = [];
          for (const message of bodyJson.messages) {
            if (
              message?.role === "assistant" &&
              Array.isArray(message.content)
            ) {
              const filteredContent = message.content.filter(
                (item) => item && item.type !== "thinking",
              );
              if (filteredContent.length !== message.content.length) {
                removedThinking = true;
                message.content = filteredContent;
              }
              if (message.content.length === 0) {
                removedThinking = true;
                continue;
              }
            }
            filteredMessages.push(message);
          }
          if (removedThinking) {
            if (VERBOSE) {
              log("Removing assistant thinking blocks");
            } else {
              info("🧹 Removing assistant thinking blocks");
            }
            bodyJson.messages = filteredMessages;
            modifiedBody = Buffer.from(JSON.stringify(bodyJson));
            modelRewritten = true;
          }
        }

        // Ensure tool_use/tool_result pairs stay in sync
        if (Array.isArray(bodyJson.messages)) {
          const originalLength = bodyJson.messages.length;
          const sanitizedMessages = [];
          let removedToolPairs = false;

          for (let i = 0; i < bodyJson.messages.length; i += 1) {
            const message = bodyJson.messages[i];
            if (
              message?.role === "assistant" &&
              Array.isArray(message.content)
            ) {
              const toolUseIds = message.content
                .filter((item) => item && item.type === "tool_use" && item.id)
                .map((item) => item.id);

              if (toolUseIds.length > 0) {
                const nextMessage = bodyJson.messages[i + 1];
                const hasMatchingToolResult =
                  nextMessage?.role === "user" &&
                  Array.isArray(nextMessage.content) &&
                  nextMessage.content.some(
                    (item) =>
                      item &&
                      item.type === "tool_result" &&
                      toolUseIds.includes(item.tool_use_id),
                  );

                if (!hasMatchingToolResult) {
                  message.content = message.content.filter(
                    (item) => item && item.type !== "tool_use",
                  );
                  removedToolPairs = true;
                }
              }

              sanitizedMessages.push(message);
              continue;
            }

            if (message?.role === "user" && Array.isArray(message.content)) {
              const prevMessage = bodyJson.messages[i - 1];
              const validToolUseIds = new Set(
                prevMessage?.role === "assistant" &&
                Array.isArray(prevMessage.content)
                  ? prevMessage.content
                      .filter(
                        (item) => item && item.type === "tool_use" && item.id,
                      )
                      .map((item) => item.id)
                  : [],
              );

              const hadToolResults = message.content.some(
                (item) => item && item.type === "tool_result",
              );

              if (hadToolResults) {
                const filteredContent = message.content.filter(
                  (item) =>
                    item &&
                    (item.type !== "tool_result" ||
                      (item.tool_use_id &&
                        validToolUseIds.has(item.tool_use_id))),
                );
                if (filteredContent.length !== message.content.length) {
                  removedToolPairs = true;
                  message.content = filteredContent;
                }
              }

              if (message.content.length === 0) {
                removedToolPairs = true;
                continue;
              }

              sanitizedMessages.push(message);
              continue;
            }

            sanitizedMessages.push(message);
          }

          if (sanitizedMessages.length !== originalLength || removedToolPairs) {
            if (VERBOSE) {
              log("Sanitizing tool_use/tool_result pairs");
            } else {
              info("🧹 Sanitizing tool_use/tool_result pairs");
            }
            bodyJson.messages = sanitizedMessages;
            modifiedBody = Buffer.from(JSON.stringify(bodyJson));
            modelRewritten = true;
          }
        }

        // Check if model needs rewriting
        if (bodyJson.model) {
          originalModel = bodyJson.model;

          // Check if it matches any known pattern
          for (const [pattern, bedrockModel] of Object.entries(MODEL_MAP)) {
            if (
              bodyJson.model.includes(pattern) ||
              bodyJson.model === pattern
            ) {
              newModel = bedrockModel;
              bodyJson.model = bedrockModel;
              modifiedBody = Buffer.from(JSON.stringify(bodyJson));
              modelRewritten = true;

              // Simple log for non-verbose mode
              if (pattern.includes("haiku")) {
                info("🔀 Intercepting haiku → bedrock haiku");
              } else if (pattern.includes("sonnet")) {
                info("🔀 Intercepting sonnet → bedrock opus");
              }

              break;
            }
          }

          if (VERBOSE && modelRewritten) {
            log(`Rewriting model from "${originalModel}" to "${newModel}"`);
          } else if (!modelRewritten) {
            if (VERBOSE) {
              log(`Model not rewritten: "${originalModel}"`);
            } else {
              info(`ℹ️ Model passthrough: ${originalModel}`);
            }
          }
        }
      } catch (e) {
        log("Failed to parse body as JSON:", e.message);
      }
    }

    if (LOG_REQUESTS) {
      const timestamp = new Date().toISOString();
      const safeTimestamp = timestamp.replace(/[:.]/g, "-");
      requestCounter += 1;
      requestId = `${safeTimestamp}-${process.pid}-${requestCounter}`;
      const logEntry = {
        type: "request",
        requestId,
        timestamp,
        method: req.method,
        url: req.url,
        cleanPath,
        headers: req.headers,
        originalBody: parsedBodyJson ?? bodyStr,
        forwardedBody: safeParseJson(
          modifiedBody.length > 0 ? modifiedBody.toString() : "",
        ),
      };
      requestLogEntry = logEntry;
      writeJsonlLog(requestLogEntry, requestId);
    }

    // Log original headers (verbose only)
    if (VERBOSE) {
      log("Original headers:", JSON.stringify(req.headers, null, 2));
    }

    // Prepare headers for forwarding (strip problematic ones)
    const forwardHeaders = { ...req.headers };

    // Remove beta-related headers
    delete forwardHeaders["anthropic-beta"];

    // Update host header
    forwardHeaders.host = TARGET_HOST;

    // Update content-length if body was modified
    if (modifiedBody.length !== body.length) {
      forwardHeaders["content-length"] = modifiedBody.length;
    }

    if (VERBOSE) {
      log("Forwarding headers:", JSON.stringify(forwardHeaders, null, 2));
      log("Clean path:", cleanPath);
    }

    // Forward request to LiteLLM
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
        if (VERBOSE) {
          log(`Response status: ${proxyRes.statusCode}`);
        }

        responseStarted = true;

        // Forward response headers
        res.writeHead(proxyRes.statusCode, proxyRes.headers);

        // Stream response back
        proxyRes.pipe(res);

        if (LOG_REQUESTS) {
          let responseChunks = [];
          proxyRes.on("data", (chunk) => {
            responseChunks.push(chunk);
          });

          proxyRes.on("end", () => {
            const responseText = Buffer.concat(responseChunks).toString();
            writeJsonlLog(
              {
                type: "response",
                requestId,
                timestamp: new Date().toISOString(),
                statusCode: proxyRes.statusCode,
                headers: proxyRes.headers,
                body: safeParseJson(responseText),
              },
              requestId,
            );
          });
        }

        // Log response for debugging
        if (VERBOSE) {
          let responseBody = [];
          proxyRes.on("data", (chunk) => {
            responseBody.push(chunk);
          });

          proxyRes.on("end", () => {
            const fullResponse = Buffer.concat(responseBody).toString();
            if (proxyRes.statusCode >= 400) {
              log("Error response:", fullResponse);
            }
          });
        }
      });

      proxyReq.on("error", (err) => {
        const shouldRetry =
          !responseStarted && attempt <= MAX_RETRIES && isRetryableError(err);

        if (shouldRetry) {
          const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1);
          if (VERBOSE) {
            log(
              `Proxy request error (${err.code}); retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`,
            );
          }
          setTimeout(sendProxyRequest, delay);
          return;
        }

        console.error("Proxy request error:", err);
        if (!res.headersSent) {
          res.writeHead(502, { "Content-Type": "application/json" });
        }
        if (!res.writableEnded) {
          res.end(
            JSON.stringify({ error: "Proxy error", details: err.message }),
          );
        }
      });

      // Send the body (possibly modified)
      if (modifiedBody.length > 0) {
        proxyReq.write(modifiedBody);
      }

      proxyReq.end();
    };

    sendProxyRequest();
  });
});

server.listen(PROXY_PORT, () => {
  console.log(`🔧 Proxy server listening on http://localhost:${PROXY_PORT}`);
  console.log(`📡 Forwarding to https://${TARGET_HOST}:${TARGET_PORT}`);
  console.log(`🧹 Stripping anthropic-beta headers and ?beta query params`);
  if (LOG_REQUESTS) {
    console.log(`🗃️  Request logging enabled: ${LOG_DIR}`);
  }
  console.log(`🗺️  Model mappings:`);
  for (const [from, to] of Object.entries(MODEL_MAP)) {
    console.log(`   ${from} → ${to}`);
  }
  if (VERBOSE) {
    console.log(`📢 Verbose mode enabled`);
  } else if (SUMMARY) {
    console.log(`🧾 Summary mode enabled`);
  } else {
    console.log(`💡 Run with --verbose or -v for detailed logs`);
  }
  console.log();
});
