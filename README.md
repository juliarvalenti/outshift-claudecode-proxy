# outshift-claudecode-proxy

A lightweight HTTP proxy for Claude Code that forwards requests to LiteLLM, strips beta flags, and normalizes request payloads so they work with Bedrock-backed model groups.

## Requirements
- Node.js 18+ (or any recent LTS that supports `https` and modern JS syntax)

## Run the proxy
From this repo:

    ./proxy-no-beta.js
    # or
    node proxy-no-beta.js

Optional flags:

    --verbose   Print full request/response details
    --summary   Print a brief message summary
    --log-requests   Write request/response JSONL logs

The proxy listens on `http://localhost:8099`.

## Configure Claude Code

Update `~/.claude/settings.json` to point Claude Code at the local proxy:

    {
      "env": {
        "ANTHROPIC_BASE_URL": "http://localhost:8099/",
        "ANTHROPIC_AUTH_TOKEN": "YOUR_TOKEN_HERE",
        "ANTHROPIC_MODEL": "bedrock/global.anthropic.claude-opus-4-6-v1"
      }
    }

Notes:
- Keep `ANTHROPIC_AUTH_TOKEN` private.
- If you prefer a different model mapping, update `DEFAULT_MODEL` and `MODEL_MAP` in `proxy-no-beta.js`.

## What the proxy does
- Strips the `anthropic-beta` header and `?beta` query params.
- Removes unsupported fields:
  - `context_management`
  - `system[].cache_control`
  - `tools[].defer_loading`
  - `tools[].custom.defer_loading`
- Fixes `thinking.budget_tokens` if it is `>= max_tokens`.
- Rewrites models via `MODEL_MAP`.

## Model mapping
Model rewriting is controlled by `DEFAULT_MODEL` and `MODEL_MAP` in `proxy-no-beta.js`. Update these if your LiteLLM model groups differ.