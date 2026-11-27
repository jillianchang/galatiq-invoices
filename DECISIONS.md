# Design Decisions

Short records of each significant decision: context, choice, and why. Alternatives are noted where they were seriously considered.

---

## D1. Hybrid extraction: parsers for structured formats, LLM for unstructured

**Context.** Invoices arrive as JSON, CSV, XML, TXT, PDF, and email bodies.
**Decision.** JSON/CSV/XML use deterministic parsers; TXT/PDF/email use Grok structured extraction. Structured files fall back to the LLM if parsing fails.
**Why.** Parsers are free, instant, and can't hallucinate. The LLM is used where it adds value.
**Validation.** A v2 eval compares LLM-only vs. hybrid extraction on accuracy, cost, and latency, so the choice is backed by data.

## D2. Self-correction via grounding, not self-assessment

**Decision.** After extraction, every number and item must appear in the source text (with fuzzy matching for OCR noise), and the line math must reconcile. Failures go back to Grok as specific errors, up to 3 retries.
**Why.** Checking output against the source is far more reliable than asking a model whether it was right.
**Math is not a grounding failure.** Real invoices have arithmetic errors (1007, 1013). If every value is grounded but the math doesn't reconcile, the model gets one re-read (to catch a missed line) and is told not to fix the vendor's math; the values are then accepted as written and validation flags the mismatch. Otherwise a vendor's mistake would become `extraction_failed` instead of `total_mismatch`.
**Dates stay raw.** The model copies date strings verbatim; parsing them is deterministic, so a relative date like "yesterday" is never turned into a guessed calendar date.

## D3. Agents can escalate, never de-escalate

**Decision.** LLM agents may add findings, decide within limits, and explain. Deterministic code owns hard constraints and payment. An LLM can only make the system more cautious.
**No LLM item matching.** Item names map to SKUs only by deterministic normalization (case, spacing, parentheticals). An LLM fallback was planned, but it would be the one place an LLM could remove a block (turning an unknown item into a known one), the provided data never needs it, and a plausible-looking wrong match (`WidgetC` → `WidgetA`) would silently clear 1016's hold. Unknown items stay unknown for a human.
**Actions come from codes.** Agents choose a finding code from a fixed list; the action is fixed by the code registry, so a model can't pick a softer action.
**Fraud rejection needs corroboration.** An LLM `suspected_fraud` finding rejects only when another independent red flag is present (payment pressure, injection, vendor change, implausible address, no catalog items, unparseable due date). Uncorroborated, it is held for a human. On Groq, gpt-oss-20b called the White House address on 1005 fraud on its own; one odd detail shouldn't reject a real vendor. The same rule binds the approval stage: a HOLD that only LLM findings block can't be decided REJECT, so the approval agent and critic can't get around it (on Groq they escalated 1004 from HOLD to REJECT over its address alone).
**An implausible address alone doesn't block.** `suspicious_vendor_address` holds only with another red flag; alone it is a WARN, shown to the approval agent and critic, with the address check in the recommended action. 1004 (742 Evergreen Terrace) is one of the spec's normal orders, and `invoice_1004_revised.json` only makes sense if 1004 is paid; 1005 (the White House) is held for stock anyway.
**Evidence must be grounded.** Every investigator finding quotes the document; a quote that doesn't appear in the document or invoice data drops the finding.

## D4. Fixed graph orchestration, no LLM supervisor

**Decision.** LangGraph with conditional edges on structured state. Agency lives inside bounded nodes (extraction loop, investigator tool use, approval agent + critic).
**Why.** Payment workflows need auditability and reproducibility. An LLM choosing control flow adds risk without adding value here.

## D5. Gate constraints and final outcomes

**Decision.** A deterministic policy gate computes one constraint from the findings: DUPLICATE > REJECT (fraud or invalid data) > RETURN_TO_VENDOR (invoice is wrong) > HOLD (internal issue) > ELIGIBLE. Final statuses are PAID, ON_HOLD, RETURNED_TO_VENDOR, REJECTED, or DUPLICATE_BLOCKED.
**Why.** The finer statuses map to genuinely different actions in accounts payable (a stock overage might resolve when goods arrive; a math error requires the vendor to resend). They still reduce to the spec's two outcomes: approved and paid, or not approved and logged with reasoning.

## D6. Validation finds; approval decides

**Context.** The spec places decision-making in the approval stage: a simulated VP review with rule-based decisions, where the agent reasons through approval or rejection with a reflection or critique loop.
**Decision.** Validation only produces findings. The approval stage makes every decision: the policy gate sets hard constraints, then an approval agent acting as the simulated VP reasons through every non-duplicate invoice within them. It has real authority on ELIGIBLE invoices (approve or escalate), recommends resolutions for HOLDs, and writes the reasoning for rejections. A critic challenges it whenever the invoice is over $10K or has any finding.
**Critic scope.** The spec asks for a critique loop on approval *and* rejection, so the critic reviews every non-duplicate decision that `critic_required_when` selects, but its power depends on the gate. On ELIGIBLE it can push APPROVE to HOLD; on HOLD it can push to REJECT; on RETURN_TO_VENDOR it can push to REJECT and checks that the vendor message states each discrepancy accurately; on REJECT the decision is fixed and it checks only that the reasoning cites the findings that drive it. It can never move a decision toward payment: a REJECT or RETURN gate comes from a deterministic check, and letting an argument (or injected invoice text) overturn it would make the LLM the path to money (D3). Duplicates skip the critic; blocking one is a database fact. Unresolved objections turn an APPROVE into HOLD, flag a vendor message for review before sending, and are recorded on a rejection without changing it.
**Enforced in code.** Each gate has a fixed set of allowed decisions; an answer outside it is replaced by the gate's fail-closed default (ELIGIBLE becomes HOLD). If the LLM is unavailable, the same default applies with deterministic reasoning, so a run with no LLM never pays anything.
**Superseded.** An earlier version had validation decide routes and made the LLM advisory to a human approver. That left clean invoices and rejections with no approval reasoning, which doesn't match the spec.

## D7. Single-shot CLI with the agent as default approver

**Context.** The spec defines `python main.py --invoice_path=...` and says approvals are simulated. Real approval is asynchronous.
**Decision.** The spec's command always completes end to end with no prompts. `--approver=agent` (default) has the approval agent act as the simulated VP. `interactive` (v2) lets a human override holds; `queue` (v2) leaves holds pending for the Streamlit approval inbox.
**Why.** A grader running the spec's command must never be stopped by a prompt. Human-in-the-loop and the async lifecycle remain as extensions that address Acme's real pain (VP approval via email chains).

## D8. Persistence in `inventory.db`

**Decision.** One SQLite file, using the spec's name. It holds reference data (`inventory`, read-only to the system) and operational data the system writes (`submissions`, `invoices`, `findings`, `decisions`, `payments`, `audit_events`). The spec's rejection log is a `rejection_log` view over those tables, not a separate table, so it can never disagree with the decisions it summarizes. `audit_events` holds one compact row per pipeline step so the dashboard can show an invoice's timeline with a query; full LLM detail (prompts, tokens, retries) stays in the JSONL trace files (D20) rather than being duplicated. `--reset` re-seeds and clears history.
**Why.** The provided data includes cross-file duplicates and a revision. Run file by file, the only way to catch them is to remember prior runs. The spec already persists the database to a file.

## D9. Duplicates, reruns, and revisions: always run the full pipeline

**Context.** 1011, 1012, and 1013 are the only PDFs and each has a TXT or JSON twin. They may exist to test PDF ingestion, not duplicates. Stopping early on a duplicate would hide the PDF pipeline from a grader.
**Decision.** Every run executes every stage. Three cases are distinguished:
- Same raw content as a prior submission (a rerun): processed normally; payment skipped by idempotency with an explicit message.
- Different submission, same normalized invoice: `duplicate` finding; output shows what the gate would have been, then that payment is blocked.
- Same vendor and number, different content: revision handling (hold with delta if the original was paid; supersede if unpaid; process normally if the original was returned to the vendor).

The fingerprint is normalized vendor + invoice number, excluding the total, so a duplicate with an altered amount can't slip through. Content comparison uses normalized fields, not raw bytes, so PDF and TXT versions match. When a result depends on earlier runs, the output says so.
**Alternatives.** Stopping at DUPLICATE (hides the pipeline); batch-only detection (misses duplicates in the spec's own CLI mode; unrealistic); showing a cached result on rerun (looks like nothing ran).

## D10. Stock semantics: per invoice, aggregated per SKU

**Context.** In real AP this is a three-way match (PO, goods receipt, invoice). Without PO or receipt data, `stock` stands in for the quantity received.
**Decision.** `stock` is modeled as the quantity available to invoice against. Quantities are summed per SKU within an invoice, but stock is not consumed across invoices.
**Why.** Aggregating catches 1013 (every line passes, totals don't). Consuming across a batch would make 1001, 1004, and 1006 together exceed WidgetA stock, contradicting the spec's expected outcomes.

## D11. No vendor master table

**Context.** There is no ground truth for vendor data; any table would be invented.
**Decision.** Detect vendor problems only from evidence in the invoice (missing name, rename notes, payment-pressure language, investigator judgment).
**Why.** A fabricated vendor table would mean choosing the answers and grading against them. Production would validate against Acme's ERP vendor master.
**Consequence.** Planted vendor signals like 1005's address (1600 Pennsylvania Ave) are the investigator's job: it raises `suspicious_vendor_address` with cited evidence. That is judgment, not a lookup, so it is marked [llm] in the evals.

## D12. Catalog prices inferred from the data; flag overcharges only

**Decision.** Add a `unit_price` column (the spec explicitly allows extending the seed) with WidgetA $250, WidgetB $500, GadgetX $750, which are consistent across every clean invoice. FakeItem gets a NULL price (no basis to infer one), and the price rule skips items without a catalog price. Flag prices over catalog; ignore discounts. No other items are added, so the spec's unknown-item scenarios stay intact.
**Why.** Prices are well supported by the data, unlike vendor details. Underpricing doesn't harm Acme.

## D13. Approval threshold means scrutiny, not an automatic hold

**Decision.** $10,000, from the spec's example, configurable in `policy.yaml`. Over threshold → the critic loop is mandatory. Any APPROVE the critic reviews (over threshold or with any finding) stands only if the loop ends with no unresolved objections; otherwise it becomes HOLD (`approve_requires_no_unresolved_objections`). Within 3% below → WARN.
**Why.** The spec asks for "additional scrutiny," not mandatory human approval. A single invoice just under the line is weak evidence, so it warns rather than holds.

## D14. Math mismatch → return to vendor with the discrepancy

**Decision.** Any arithmetic mismatch sets the gate to RETURN_TO_VENDOR, and the reasoning states stated vs. computed values.
**Why.** The vendor owns the invoice's correctness; paying a recomputed amount silently creates reconciliation problems.

## D15. Record/replay instead of a mock LLM

**Decision.** LLM calls are cached by hash of (model, messages, schema). `record` saves real Grok responses; `replay` runs offline from the committed cache.
**Why.** Reviewers can run everything without a key, tests are deterministic, and outputs are real model outputs. A hand-written mock would test nothing about the LLM.
**Note.** The cache is keyed by model, so the committed cache is recorded on Grok (see D23).

## D16. Fail closed

**Decision.** LLM timeouts, malformed output after retries, or unparseable documents → `extraction_failed` HOLD.
**Why.** A payment system must never default to approval when it is uncertain.

## D17. Deterministic text-signal backstop

**Decision.** Keyword rules for vendor identity change, payment pressure, and prompt injection (bank-detail change in v2) run regardless of the LLM. The investigator adds judgment on top.
**Why.** Defense in depth: the safety floor holds even if the LLM misses something or is unavailable.
**Fraud pattern.** A deterministic `fraud_pattern` REJECT fires when a zero-stock item is billed together with payment pressure on an over-threshold invoice. The spec's headline fraud case (1003) therefore never depends on the LLM to be rejected.

## D18. Terms tolerance

**Decision.** Due date vs. invoice date + payment terms mismatches warn only beyond 3 days. No reference "today" date is used; relative dates like "yesterday" are treated as unparseable.
**Why.** Small terms discrepancies are noise.

## D19. Non-USD currency

**Decision.** Convert via a static FX table (no internet) for display and price checks, and always HOLD.
**Why.** FX introduces judgment (rate, date, fees) that a human should confirm in v1.

## D20. Output, tracing, and money

**Decision.** By default the console shows one structured JSON log line per stage, a readable results panel, and the final result as JSON; `--json` prints JSON only. Full traces go to local JSONL files instead of LangSmith. All money uses `Decimal` with a $0.01 tolerance.
**Why.** The spec asks for structured logs and results. The runtime assumes no internet. Float arithmetic on currency is a correctness bug.

## D21. Expected outcomes are tagged by source

**Decision.** Each entry in `expected.yaml` is tagged `source: spec` (the spec defines its validation result: 1001–1004, 1006, 1008, 1009, 1016) or `source: assumption` (our judgment). Outcomes are always our policy. Evals run standalone (each file on a fresh DB) and batch (all files in sorted order).
**Why.** Otherwise "the evals pass" is circular. Standalone runs mirror how a grader uses the spec's command; batch runs exercise duplicates and revisions.

## D22. Tiered scope

**Decision.** v1 (~9.5h, including the Grok swap) is the spec's four stages end to end with the gate, approval agent + critic, idempotent payment, structured logs, record/replay, evals, a read-only Streamlit dashboard, and README. v2 is the Streamlit approval queue, extra approver modes, most extra test cases, and comparisons. Some items are README-only.
**Why.** The spec says "a few hours to a day" and grades shipping an MVP under ambiguity with scope cut ruthlessly. The dashboard is in v1 because UI/UX is a graded criterion; the clean over-$10K test case is in v1 because no provided invoice exercises the over-threshold approval path.

## D23. Develop on Groq, ship on Grok

**Context.** The spec makes Grok the core reasoning engine but allows other models if no key is available. Groq has a free tier for the open-weight gpt-oss models. Both offer OpenAI-compatible APIs.
**Decision.** One client on the OpenAI Python SDK, with the provider chosen by `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and an optional `LLM_MODEL_FAST`. Development and iteration run on Groq's free tier: gpt-oss-20b for extraction and investigation, gpt-oss-120b for approval and the critic, reasoning effort low. A Grok smoke test runs early; final evals, prompt tuning, and the committed replay cache are on Grok. The README reports Grok as the headline result with a small Groq (gpt-oss) vs. Grok comparison.
**Why.** Keeps development free while demoing on the engine Galatiq builds on. Prompts can behave differently across models, so a dedicated swap phase is budgeted. Compatible APIs can differ on strict schemas, tool calling, and errors, so Pydantic validation and grounding remain the source of truth on any provider. Groq's free tier allows about 50 calls per model per day against ~220 for a full eval, so development iterates on a few invoices at a time with the replay cache, and full evals run on Grok. Splitting roles by model uses both daily allowances and puts the stronger model where judgment matters.
