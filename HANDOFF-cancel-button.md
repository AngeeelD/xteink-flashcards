# Handoff — Cancel button for running jobs

> **Status**: ready for a separate session, planned with gentle-ai RDD (native review).
> **Repo**: https://github.com/AngeeelD/xteink-flashcards (branch `main`)
> **Picked up from**: the delete-job carousel work in `52daee4`.

---

## 1. Context (where the project is right now)

`webapp/app.py` runs each uploaded epub through a `threading.Thread` daemon
(`_run_job`, line ~526). The thread:

1. Reads the saved epub under `webapp/uploads/<job_id>.epub`.
2. Lists chapters with `list_chapter_files`.
3. Phase A: `_generate_bilingual_per_chapter` writes per-chapter CSVs into
   `webapp/jobs/<id>/csv/`, zips them into `flashcards.zip`.
4. Phase B: `_enrich_per_chapter` calls Ollama + StarDict and writes
   `webapp/jobs/<id>/enriched/chapter_NNN.csv`.
5. Phase B2: `_render_bmps_per_chapter` reads the enriched CSVs and renders
   BMPs into `webapp/jobs/<id>/bmp/` (light) or `bmp_dark/` (dark), then zips
   into `screensaver.zip` / `screensaver_dark.zip`.
6. Updates `meta.json` with status (`pending` → `running` → `done` / `error`).

`_active_job_id` is a module-level string that gates the single-job lock.
The job page (`/job/<id>`) polls `/job/<id>/status.json` every 2.5 s and
shows the current phase, counters and download buttons when done.

The recent-jobs carousel (`_recent_jobs`, line ~827) shows the last six
**done** jobs; each card has a small × that pops a `confirm()` dialog and
POSTs to `/job/<id>/delete` (`_delete_job`, line ~801) which
`shutil.rmtree`s the job directory.

There is currently **no way to stop a running job**. The worker blocks on
`ollama_generate_bi(...)` and `ef_ollama_generate(...)` for tens of seconds at
a time. If the user realises they picked the wrong epub, wrong chapter
range, or just changed their mind, they have to wait for the current
chapter to finish (or `docker compose restart` the whole thing, which
loses state and breaks the running-card UI).

---

## 2. Why this is a separate effort

The user explicitly said (in the conversation that landed `52daee4`):

> *"un cancel fuerte para poder liberar recursos del equipo, si quieres
> esto lo gestionamos en un esfuerzo totalmente separado, igual lo hacemos
> con RDD de gentle-ai"*

So this is **not** a small follow-up commit. Threading + Ollama
cancellation + UI + cleanup + tests is at least four work units and needs
native review on the checkpoint logic and the Ollama-call cancellation
path. Keep this handoff self-contained so a fresh session can pick it
up without scrolling the chat history.

---

## 3. Problem statement (RDD)

### Functional requirements

- **R1 — Cancel button visible only while running.** A "Cancel job"
  affordance appears on `/job/<id>` only when `meta.status ∈ {pending,
  running, queued}`. Done / error / cancelled jobs don't get the button.
- **R2 — Confirmation dialog.** Clicking Cancel pops a `confirm()` (or
  custom modal — same dialog system the delete button uses). The message
  must list what gets removed (partial BMPs, enriched CSVs) and that the
  action can't be undone.
- **R3 — Strong cancel, frees resources.** Cancel must actively stop the
  worker, not just mark it as cancelled and wait. In particular it must
  release the Ollama slot: a hung `ollama_generate` call must not keep
  holding the HTTP connection after the user clicks cancel.
- **R4 — Mark as cancelled in meta.json.** Status flips to `cancelled`
  with a `cancelled_at` ISO timestamp and a `cancelled_chapters` count
  (how many chapters completed before the cancel).
- **R5 — Clean up partial outputs.** After cancellation, the job
  directory contains only `meta.json` and the original epub upload (if
  we decide to keep it). `bmp/`, `bmp_dark/`, `enriched/`, `csv/` and any
  partial `*.zip` are removed. `_zip_dir` already handles empty dirs; we
  just need to point it at cleaned folders.
- **R6 — Free the active slot.** `_active_job_id` is cleared (under the
  job lock) so the next upload can take over without a 409.
- **R7 — Idempotent.** Hitting cancel twice in a row is a no-op the
  second time. Cancelling a job that's already `done` is a 409 (or 404).
- **R8 — UI reflects state.** The job page polls `/status.json` every
  2.5 s; once `status === "cancelled"`, the page stops the spinner and
  shows a "Cancelled" badge + the cleanup timestamp. The active job tile
  on `/` disappears (it's no longer running) but the job is still
  accessible at `/job/<id>` for inspection.

### Non-functional

- **N1 — Latency.** The worker should pick up the cancel within 5 s in
  the worst case (one full Ollama call ≈ 10–30 s is acceptable as an
  upper bound; if we can't break the in-flight call, at least the next
  checkpoint stops).
- **N2 — No data loss for completed chapters.** Chapters that already
  finished before cancel keep their enriched CSV / BMPs in a "partial"
  subdir if we want to expose them, OR we wipe them per R5. Pick one and
  document.
- **N3 — No Flask crashes.** Killing the worker must not corrupt the
  meta.json, leave dangling locks, or break subsequent uploads.

### Non-goals (explicitly out of scope)

- **Resumable jobs.** No "resume from chapter N" — clean cancel only.
- **Pause / suspend.** No temporary pause. If the user wants to pause,
  they cancel and re-upload later.
- **Per-chapter cancel granularity.** One cancel = whole job. If we want
  per-chapter later it's a follow-up.
- **Custom modal dialogs (HTML/CSS).** Use `confirm()` for now, same as
  delete. Modal polish is a separate UI effort.

---

## 4. Technical context (what the implementer needs to know)

### Worker architecture

```
POST /upload  ──►  threading.Thread(target=_run_job, args=(job_id, cfg), daemon=True)
                       │
                       ▼
                  _run_job(job_id, cfg)
                       │
                       ├── _generate_bilingual_per_chapter(...)   # CSV gen
                       ├── _enrich_per_chapter(...)               # Ollama + StarDict
                       ├── _render_bmps_per_chapter(...)           # BMP render
                       └── meta["status"] = "done"
```

- All three generators iterate over chapters with `enumerate(..., start=1)`
  and call `update_status(...)` between iterations — those are the
  natural cancellation checkpoints.
- The longest blocking call inside the loops is
  `ef_ollama_generate(prompt, OLLAMA_MODEL, host=OLLAMA_HOST)` from
  `enrich_flashcards.py`. That's the call that needs to be cancellable.

### Ollama call shape

```python
# scripts/enrich_flashcards.py — typical Ollama call
def ef_ollama_generate(prompt: str, model: str, host: str = "http://localhost:11434",
                      stream: bool = False) -> str:
    response = requests.post(f"{host}/api/generate", json={...})
    return response.json()["response"]
```

It's a **blocking** `requests.post` — no timeout, no stream. To cancel it
we'd need to:
- Switch to `stream=True` and read chunks in a loop (so we can break
  the read on cancel).
- OR run the call in a thread, expose a cancel flag the thread polls
  every 100 ms, and `requests.Session.close()` from outside.
- OR use Ollama's documented "load model → unload model" lifecycle to
  cancel in-flight generations (research needed).

Native review should pick the cleanest option; don't lock it in here.

### State to track

- `webapp.app._active_job_id: str | None` (already exists).
- A new `webapp.app._cancelled_jobs: set[str]` or per-job
  `threading.Event`. Recommendation: **per-job Event** — cleaner
  scoping, automatic cleanup when the worker thread ends.

```python
# in _run_job:
cancel_event = threading.Event()
_cancelled_jobs[job_id] = cancel_event
try:
    ...
finally:
    _cancelled_jobs.pop(job_id, None)
```

### Files to touch (rough)

| File | What |
|---|---|
| `webapp/app.py` | New endpoint, cancel registry, checkpoint checks in `_run_job` / `_enrich_per_chapter` / `_render_bmps_per_chapter` / `_generate_bilingual_per_chapter`, cleanup helper. |
| `scripts/enrich_flashcards.py` | Make `ef_ollama_generate` cancellable (stream + break, or thread + close). |
| `scripts/epub_to_flashcards.py` | Same treatment for `ollama_generate_bi`. |
| `webapp/templates/job.html` | Cancel button + confirm() + status badge for `cancelled`. |
| `webapp/static/style.css` | Cancel button styling (probably matches the existing `button[role="button"]` palette). |
| `tests/test_webapp.py` | Cancel endpoint, checkpoint detection, cleanup, idempotency. |
| `tests/test_flashcard_pipeline.py` | Maybe: cancel signal interrupts enrichment between chapters. |

### What already exists (don't redo)

- `_delete_job` (line ~801) — already does the recursive rmtree. Reuse
  the same pattern for cancel cleanup (or factor `_cleanup_partial_outputs`).
- The job lock (`_job_lock`) — use it when mutating `_active_job_id` and
  the cancel registry.
- `update_status(**kw)` — the worker already uses it per chapter. The
  cancel check should sit right next to it in each loop.
- The 2.5 s status polling on `job.html` — the UI already re-renders
  when `status` changes. `cancelled` will flip automatically.

---

## 5. Acceptance criteria (what "done" looks like)

A native-review-friendly checklist:

- [ ] `POST /job/<id>/cancel` returns 200 for a running job, 409 for a
      done/errored job, 404 for a missing id.
- [ ] After cancel, `meta.json` reads `status=cancelled`, has
      `cancelled_at` (ISO UTC) and `cancelled_chapters` (int).
- [ ] Worker stops within 5 s in the simple case (no in-flight Ollama).
- [ ] Worker stops within 30 s worst case (one full Ollama call).
- [ ] After cancel, the job directory contains only `meta.json` (plus
      optionally the original epub at `webapp/uploads/<id>.epub`, which
      is a separate dir).
- [ ] `_active_job_id` is cleared (next upload returns 202, not 409).
- [ ] Cancel button is hidden for `status ∈ {done, error, cancelled}`.
- [ ] Hitting cancel twice doesn't crash or duplicate work.
- [ ] The job page polls and shows "Cancelled" with the timestamp
      without manual refresh.
- [ ] Existing tests still pass (74 + the new ones).
- [ ] New tests cover each requirement above.

---

## 6. Risks / things to surface to native review

- **Race between cancel and a finished chapter.** The worker may be
  mid-write of an enriched CSV when cancel fires. Decide whether the
  in-flight write completes (then cleanup removes it) or gets
  half-truncated. Prefer "let the write finish, then cleanup" — simpler
  reasoning.
- **Ollama HTTP keep-alive.** A blocking `requests.post` can hold a
  socket open. Cancelling the call should `Session.close()` (or move to
  streaming with explicit break) to free the FD.
- **Concurrency with the job lock.** If we wrap `_active_job_id =
  None` in the lock, the next upload can take over. If we don't, the
  next upload sees stale state. Use the lock.
- **Native-review-style "is the cancel actually observable?" check.**
  A test that asserts `cancel_event.is_set()` becomes true within N ms
  of POSTing cancel — easier to assert than timing the actual stop.
- **The `_run_job` finally block already clears `_active_job_id` when
  the job finishes naturally.** Make sure cancel uses the same finally
  block so the active-id bookkeeping is consistent.

---

## 7. Open questions for the next session

- [ ] Cancel latency budget — is 5 s the right target, or do we want
      faster (sub-second)?
- [ ] Should cancelled jobs still show in the recent-jobs carousel?
      Currently `_recent_jobs` filters `status == "done"`. Easy to flip
      to "done or cancelled" but worth deciding explicitly.
- [ ] Should we keep the epub file at `webapp/uploads/<id>.epub` after
      cancel? Right now uploads stay forever (gitignored). Probably yes
      — it's small and useful for re-running with different settings.
- [ ] Confirmation dialog — keep `confirm()` or invest in a custom
      modal that matches the retro / e-ink design system? The delete
      button uses `confirm()`; consistency argues for keeping it.

---

## 8. Suggested work units (for the implementer / RDD plan)

1. **Cancel registry + endpoint** — `POST /job/<id>/cancel` adds to
   `_cancelled_jobs`, returns 200. Tests for idempotency and 404/409.
2. **Worker checkpoints** — `cancel_event.is_set()` check at the top of
   each chapter loop in the three generators; clean exit. Tests for
   "cancel between chapters" path.
3. **Ollama cancellation** — make `ef_ollama_generate` and
   `ollama_generate_bi` actually interruptible. This is the risky bit
   — needs native review on the threading model.
4. **Cleanup + meta update** — partial output removal + status flip.
   Reuses `_delete_job`'s rmtree pattern.
5. **UI** — Cancel button + confirm() + status badge + visibility
   rules.
6. **Tests for the unhappy paths** — race conditions, double-cancel,
   cancel during in-flight Ollama.

Each unit ≤ 1 day's work and reviewable independently.

---

## 9. References inside the repo

- `webapp/app.py`:
  - `_run_job` (~line 526)
  - `_generate_bilingual_per_chapter` (~line 144)
  - `_enrich_per_chapter` (~line 248)
  - `_render_bmps_per_chapter` (~line 467)
  - `_active_job_id` and `_job_lock` (~line 90)
  - `_delete_job` (~line 801) — pattern to reuse
  - `_recent_jobs` (~line 827) — might need `cancelled` in the filter
- `scripts/enrich_flashcards.py`: `ef_ollama_generate` — the blocking
  call that needs to become interruptible.
- `scripts/epub_to_flashcards.py`: `ollama_generate_bi` — same.
- `webapp/templates/job.html`: current job status page; needs the
  Cancel button + the "Cancelled" badge.
- `tests/test_webapp.py`: existing job-lifecycle tests (WebBilingualPipeline,
  WebBmpPipeline, Download, RecentJobs, DeleteJob) — pattern to follow.

---

## 10. Handoff metadata

- **Topic key (Engram)**: `xteink-flashcards/release-2026-08-01` (append a
  new observation under this key when starting the cancel session).
- **Original chat session**: ended at `52daee4` (delete button) on
  `main`.
- **Starting commit for the new session**: `52daee4` (or whatever
  HEAD is at session start).
- **Branch**: cut `feat/cancel-button` off `main`; native review
  runs against that branch.

When the next session opens this handoff, the first native-review
question should be: *"Does this handoff's problem statement match the
requirement? If yes, propose a RDD plan covering work units 1–6
above."*
