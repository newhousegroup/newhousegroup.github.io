# Glyph Playground v1.0

A separate static text-completion website using Labroom 2.7.1's actual Glyph BPE transformer and Glyph 1x Plus reranking. No chatbot roles, model API, training UI, server inference, database, corpus, or optimizer required. Labroom was not changed.

**Includes the published glyph-1x checkpoint at 30,000 training steps.** The public assets contain only its inference weights and BPE tokenizer: training corpus, validation text and optimizer state were removed. The model downloads automatically; local `.lrm` import remains available in Advanced settings. Public model size: approximately 20.35 MiB across six shards. Architecture: 6 blocks, width 256, 8 heads, 256-token context, BPE vocabulary 1,024.

## Files

- `dist/`: complete deployable static website; no build step.
- `dist/core.js`: model, BPE, sampling, scoring, LRM manifest and SHA-256 primitives extracted from the supplied Labroom 2.7.1 HTML.
- `dist/worker.js`: worker backend selection, checkpoint validation/loading, cancellable segmented candidate generation.
- `dist/app.js`, `index.html`, `style.css`: editor, generated-text highlighting, controls, loading and settings.
- `dist/vendor/`: pinned TensorFlow.js 4.22.0 and WASM backend binaries. These are runtime dependencies, not third-party model weights.
- `dist/models/model.json`: small public manifest locating versioned checkpoint shards.
- `tools/publish_model.py`: standard-library-only Python checkpoint stripping and sharding tool.
- `tests/`: pure primitive checks, publisher checks and real TensorFlow CPU inference checks.

## Publish your trained model

1. In Labroom, open the trained **glyph-1x BPE** checkpoint you want to publish. Plus shares these weights: there is no separate Plus checkpoint to train or export.
2. Export `.lrm`. Prefer disabling both corpus and optimizer inclusion. The tool below strips them even when present. Use Labroom's desired best/latest checkpoint before exporting; this tool does not select a different weight revision.
3. In the extracted project folder, run:

```sh
python3 tools/publish_model.py /path/to/glyph-1x-25k.lrm --out dist/models
```

4. Deploy the entire `dist` directory. The page fetches `models/model.json`, downloads its shards with byte progress, verifies LRM SHA-256, validates BPE and tensor shapes, initializes weights, and enables Generate. Nothing is uploaded for inference.

The tool preserves float32 weights, tensor order, architecture, tokenizer merges and their IDs. It rebuilds an LRM1 container containing only model tensors and allowlisted inference metadata. It recomputes the payload checksum, then splits the container into **4 MiB shards**, including when a tensor crosses a shard boundary. The browser reconstructs the container before validation. This is lossless packaging, not quantization or conversion to another architecture.

The manifest is updated to content-versioned filenames. To update the model, rerun the command with a newer checkpoint and deploy. No application code changes are required. Keep older shards for a deployment transition if using non-atomic hosting; Pages deployment snapshots are atomic. Old files may be removed after old pages are no longer in use.

## Run locally

```sh
python3 -m http.server 8000 --directory dist
```

Open http://localhost:8000. Use HTTP localhost or HTTPS hosting, not a `file://` URL: workers and fetch require a proper origin. For an iPad, deploy HTTPS or serve with an appropriate trusted HTTPS development setup.

## Deploy to Cloudflare Pages

This is a buildless static site. No Worker function, API secret, GPU subscription or database is needed.

### Dashboard direct upload

1. Run the model publishing command first.
2. Sign into Cloudflare and go to **Workers & Pages** → create an application → **Pages** → **Direct Upload** (dashboard wording may vary).
3. Choose a project name, for example `glyph-playground`.
4. Upload the **contents of `dist`** as the site root, including `models`, `vendor`, `_headers`, and `index.html`. Do not upload the repository root as the website root.
5. Deploy. Open the assigned `https://<project>.pages.dev` URL. Wait for the model to report Ready, then test a prompt, Stop, Continue and Regenerate.

### Wrangler direct upload

Install a current Node.js version, then run from the project folder:

```sh
npx wrangler login
npx wrangler pages project create glyph-playground --production-branch main
npx wrangler pages deploy dist --project-name glyph-playground --branch main
```

For later updates, rerun the final deployment command. If the project already exists, skip project creation. Cloudflare authentication and the trained checkpoint are supplied by you; this package does not contain credentials.

### Git integration alternative

Connect a repository containing this project to Pages. Set the project root to this folder, framework preset to None, no build command, and build output directory to `dist`. Commit the generated model assets as well (or arrange their generation/download in your own build). Do not commit an unstripped private training checkpoint. Direct Upload and Git integration are different project workflows; choose one when creating the Pages project.

Official deployment and size references:
- https://developers.cloudflare.com/pages/get-started/direct-upload/
- https://developers.cloudflare.com/pages/platform/limits/
- https://developers.cloudflare.com/workers/wrangler/commands/pages/

Cloudflare Pages limits individual assets to 25 MiB; the model publisher's 4 MiB shards fit below that limit. Each shard is ordinary public static data.

## Exact inference behavior

The model is the same learned absolute-position, pre-layer-normalized causal transformer: learned token/position embeddings, multihead QKV attention, GELU feed-forward layers at 4x width, final layer norm, output head, and Labroom's epsilon and tensor order. Inference calls only; training is not exposed.

BPE retains Labroom's Unicode piece boundaries, UTF-8 byte vocabulary and ranked merge order. Checkpoints must be version 3 BPE LRM1. Character-level and legacy JSON/bgpt checkpoints are intentionally rejected; they are not Glyph 1x Plus.

Every segment generates 1–4 independent candidates of up to 32 tokens from the same history. It uses top-p 0.9, temperature 0.75 by default, the original repeated-ngram penalty, and the original score: likelihood × 0.55, bounded prompt lexical relevance, bounded continuity, diversity, duplicate-ngram/runs/punctuation penalties. The highest-scoring candidate is appended. Candidate count 1 still uses the same sampling path.

Temperature is now configurable; selecting 0.75 preserves Labroom's Plus temperature. The score's log probability remains the original unpenalized temperature-scaled distribution. English lexical families and stemming are preserved, including their language limitations; this is not semantic understanding or a second neural model.

After each segment is selected, its actual text is revealed word by word while inference immediately starts the next candidate group, without waiting for any UI acknowledgment. Reveal timing adapts to measured group duration (300–1,500 ms per word during inference; 300 ms after completion) to reduce empty gaps. This cannot eliminate gaps if inference is much slower than the available text. Live candidate/token progress is sent every four sampled tokens. The default candidate count is 3. Model download progress is displayed as a percentage. Unselected candidates are never shown. During the first group, a green `.`, `..`, `...` indicator animates only in the highlight overlay, never in the editable/copyable text. Once inference completes, Continue/Generate/Regenerate are available immediately even if text is still appearing; starting another run first flushes the selected tail so it is never lost. Editing and settings unlock when the reveal completes or Stop is pressed. Stop cancels the current group and immediately displays all already-selected output, including its unrevealed tail. Tokens shown count selected output tokens, not candidate evaluation work. This presentation change does not alter model sampling or ranking.

The whole document stays editable between runs. Editing makes the current document the new original prompt and resets its highlight boundary. Continue uses all current text as its prompt. Regenerate replaces only the last generation using its saved pre-generation prompt, including after Stop. Clear resets the document. During a run, editing is locked to avoid racing worker updates; press Stop to edit and resume from any point.

Long prompts initially retain the final context window for inference; the original prompt still contributes lexical scoring. At context capacity, accurate mode refills the last full window, while fast mode refills the last half-window, exactly as Labroom. Learned absolute positions restart when refilling. Neither provides unlimited attention to the full document. Empty input seeds a single byte-space as in Labroom, without adding it to the visible text.

## Devices, performance and memory

All tensors live in a dedicated worker. It tries WebGL acceleration where worker/OffscreenCanvas support exists, then single-thread WASM/SIMD, then TensorFlow's CPU backend. A small matrix operation checks initialization. No WebGPU backend is claimed or used. Unsupported worker WebGL commonly falls back to WASM, especially on mobile. Threaded WASM is disabled so cross-origin-isolation headers are not required.

The UI remains responsive while the worker computes. Stop is checked each token and yields to the worker event loop every token. An already submitted tensor operation cannot be interrupted; Stop takes effect after that operation completes. Page exit terminates the worker; returning from a browser back-forward cache reloads it. Candidate KV caches and logits are disposed in `finally`, candidates run sequentially, and only one attention mask is retained (a bounded-memory change to Labroom's mask map). Model reload terminates the previous worker before loading another model. A backend failure during generation is reported; automatic mid-run backend migration is not implemented.

Raw model weight size is approximately parameter count × 4 bytes. Browser memory also includes downloaded shards/container bytes, backend weight copies, temporary tensor allocations, attention matrices, KV caches and decoding history. Peak memory can be several times the checkpoint size; large checkpoints may exceed mobile limits. There is no reliable universal memory preflight. Reduce context/model size in your trained checkpoint, use fewer candidates or shorter output, or try a desktop if a device runs out of memory. CPU inference is a compatibility fallback and can be slow. Four candidates cost roughly four times the sampling work of one.

Pinned vendor runtime assets are included, so production inference requires only your hosting origin. See `THIRD_PARTY.md` and `dist/vendor/LICENSE` for TensorFlow.js licensing.

## Caching and public weights

`_headers` sets the small model manifest to revalidate and content-versioned model shards to one-year immutable caching. The browser uses its normal HTTP cache; there is no service worker, IndexedDB model cache, offline guarantee, or persistent storage permission. Browsers may evict cached data, especially on iPad. Each page session still initializes tensors again even when the download is cached.

Anyone with access to a public deployment can download its model weights and tokenizer. Static browser inference cannot keep them secret. Source text and manually imported checkpoints never leave the device through application code. Your hosting service still receives normal asset requests. Only publish weights/tokenizer you intend to distribute. The stripping tool removes training text and optimizer state, but a trained model can still memorize its training data.

## Verification and limits

Run from the project folder:

```sh
node --check dist/core.js
node --check dist/worker.js
node --check dist/app.js
node tests/core.test.cjs
node tests/inference.test.cjs
python3 tests/publisher.test.py
```

Checks cover Unicode BPE roundtrips/rank ordering, SHA-256 block boundaries, LRM corruption detection, actual CPU cached/full inference parity, candidate segments, both overflow modes, cancellation, stable tensor counts, complete disposal, stripping private fields, and preserving model weights. Synthetic randomly initialized test tensors are used only by tests and are never shipped as a trained model or displayed as Glyph output.

The supplied 30,000-step checkpoint is covered by an additional optional integration test (`node tests/checkpoint.test.cjs /path/to/checkpoint.lrm`) exercising real weight loading and two-candidate Plus completion. This is a functional check, not a quality benchmark. Browser visual QA, physical iPad testing, WebGL and WASM execution have not been performed here. Validate on your target browsers after publishing the model: test a Thai/English prompt, edit and resume, stop within a candidate group, regenerate, reload from cache, and compare with Labroom at matching temperature/candidates/repetition/overflow settings. Sampling is stochastic; expect distribution/behavior compatibility, not identical strings from different random streams or backends.

Missing asset/checksum/shape/backend/non-finite errors are implementation or deployment problems reported explicitly by the UI. Repetitive prose, weak factual knowledge, and lexical topic drift with otherwise valid inference can be limitations of the trained model and reranker.
