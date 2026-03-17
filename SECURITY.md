# Security Policy

## Current Status

This repository is an actively developed local-first traffic video analytics project.

It now includes:
- a computer vision pipeline
- a FastAPI web dashboard
- browser-based file upload
- local per-job artifact storage
- AI chat with Gemini-first and local fallback behavior

This means the security surface is larger than a model-training-only repository.

## Supported Usage Model

The current intended usage model is:

- local development
- local testing
- local demo hosting on a trusted machine
- controlled sharing of the codebase through GitHub

This project is **not yet positioned as a hardened public multi-tenant production service**.

## Security Controls Currently in Place

Current implemented safeguards include:

- `.env` is ignored by Git
- `storage/uploads/` is ignored by Git
- `storage/jobs/` is ignored by Git
- generated videos, CSVs, and job artifacts are excluded from version control
- upload validation exists on FastAPI entrypoints
- runtime parameter validation exists for `line_frac` and `direction`
- safer escaped HTML error rendering is used in `main.py`
- local filesystem-heavy metadata is sanitized before LLM submission
- Gemini/internal debug details are not intentionally exposed to end users
- visual scene inference is treated as unverified hypothesis, not confirmed metadata

## Secrets

### Gemini API Key

If AI chat is used, the API key should be stored only in a repo-root `.env` file:

```env
GEMINI_API_KEY=your_key_here
````

Rules:

* never commit `.env`
* never paste real keys into issues or pull requests
* rotate the key if accidental exposure is suspected

## Uploaded Files and Generated Artifacts

Uploaded traffic videos and generated outputs are stored locally under `storage/`.

Typical job artifacts may include:

* original uploaded input video
* `crossing_events.csv`
* `summary.json`
* `annotated_output.mp4`
* `status.json`
* `job_info.txt`
* `visual_inference.json`
* `chat_cache.json`

These files may contain user-provided video content and derived analytics. They should be treated as local generated data, not source-controlled assets.

## Current Risk Profile

Because the dashboard currently accepts uploads and creates local job artifacts, the main present risks are:

* accidental secret commit
* accidental commit of generated user/job data
* oversized or malformed uploads
* local resource exhaustion during video processing
* overexposure of internal metadata to third-party LLM APIs
* provider quota exhaustion for Gemini-backed chat

## Known Current Constraints

At the current stage:

* there is no authentication layer
* there is no user account system
* there is no multi-user isolation model
* there is no production-grade rate limiting
* there is no hardened cloud deployment profile yet
* local file storage is used for job results
* the intended direction runtime currently supports only `top_to_bottom`

This is acceptable for local testing and controlled demo use, but it should not be mistaken for a production SaaS security posture.

## Safe Usage Recommendations

If you are testing this project locally:

* run it on your own machine
* do not expose the local server publicly without additional hardening
* avoid using sensitive/private video content unless you accept local storage of generated artifacts
* keep your `.env` file private
* delete old `storage/jobs/` data periodically if needed
* review outbound AI usage before sharing a public demo

## Reporting a Vulnerability

If you find a security issue, please report it responsibly by contacting the repository owner directly rather than disclosing exploit details publicly first.

Include:

* a clear description of the issue
* reproduction steps
* affected file(s) or route(s)
* impact estimate
* screenshots or logs if relevant, with secrets removed

## Near-Term Security Priorities

The next security-focused improvements expected for this repo are:

* additional route-level validation and abuse controls
* more app-specific tests
* tighter deployment guidance
* safer default onboarding for public GitHub users
* continued reduction of unnecessary outbound context to LLM providers