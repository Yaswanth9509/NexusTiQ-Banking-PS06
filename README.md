TRACK_ID=PS06

# Transaction Risk Investigation Assistant

An investigation assistant for a bank's fraud desk. It takes one customer's
transaction history, reviews it against a set of risk rules, and answers the
question an investigator actually has first: **does anything here need my
attention at all?**

Most histories do not. Fifteen of the twenty sample customers come back clean,
and that is the point — a system that finds suspicion everywhere is as useless
as one that finds none.

When something does need attention, the report names the transactions involved,
says which rule was triggered, explains how the activity differs from that
customer's own established behaviour, and says what to look at first. Every
transaction it cites can be found in the input history by its identifier. It
never states that fraud has occurred; it flags, explains, and hands the
judgement to the investigator.

---

## Running it

Python 3.11 or newer is the only prerequisite. No Node, no database, no
container, no build step.

```bash
git clone https://github.com/Yaswanth9509/NexusTiQ-Banking-PS06.git
cd NexusTiQ-Banking-PS06
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:8000**. Startup takes about a second and prints:

```
Loaded 20 customer histories
Typology matching route: embeddings (cached index)
Ready on port 8000
```

That is the whole application - API and interface in one process on one port.
There is no second command, no separate frontend server, and nothing that waits
for input.

### The API key is optional

The application runs without one, and everything you need in order to judge it
works either way: the rules, the findings, the scores, the traceability and the
interface. Supply a key and each flagged customer additionally gets an
AI-written investigator briefing and embedding-based destination matching.
Without one, the AI section says plainly that it is unavailable and destination
matching falls back to keyword anchors drawn from the same document. `GET
/health` reports which mode is active, so nothing degrades silently.

```bash
export GEMINI_API_KEY=your-key        # macOS/Linux
set GEMINI_API_KEY=your-key           # Windows cmd
$env:GEMINI_API_KEY='your-key'        # PowerShell
```

A `.env` file at the root is read if present - see `.env.example`. No key is
committed anywhere in this repository.

### What to look at first

The left panel lists twenty customers. Four are worth going to directly:

| | | |
|---|---|---|
| **CUST_001** | ROUTINE | Sixty transactions including a $1,450 monthly mortgage, nothing flagged. Most customers should look like this. |
| **CUST_005** | ROUTINE | The harder case. This customer wires $5,000 to a brokerage *every month* - three times larger than anything else they do - and it is correctly left alone. Flagging it would be the same failure as missing a real one. |
| **CUST_003** | INVESTIGATE 0.73 | Three wires to a cryptocurrency exchange in three days. |
| **CUST_009** | ESCALATE 0.93 | Three rules fire at once. Read the "How these findings connect" panel: all three rest on the same three transfers, so it is one episode rather than three problems. |

**Click any transaction ID inside a finding.** It expands to show that exact row
from the source history. Every figure in every report can be traced this way,
and because the identifiers live in the committed data you can equally just
search `data/sample_customers.json` for any ID you see on screen.

### Checking it from the command line

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/customers
curl -X POST "http://localhost:8000/api/investigate?customer_id=CUST_009"
curl -X POST "http://localhost:8000/api/investigate?customer_id=CUST_001"
```

Interactive API documentation is at `http://localhost:8000/docs`.

To see the deterministic half on its own, with no model involved:

```bash
curl -X POST "http://localhost:8000/api/investigate?customer_id=CUST_009&enrich=false"
```

The verdict, score, findings and citations come back identical.

### Tests

```bash
python -m pytest tests/ -q            # 72 tests, about two seconds
```

Roughly half the rule tests assert that nothing is flagged - rent, a standing
investment transfer, a large incoming payment, a settled weekly routine. Each
was a false positive at some point and now has a test holding it shut.

To regenerate the sample data from its seed:

```bash
python -c "import sys; sys.path.insert(0,'.'); from src.data_generator import generate_sample_customers as g; import json; print(len(g()))"
```

### If something goes wrong

- **Port 8000 in use** - stop whatever holds it, or run with `PORT=8080` and open that instead.
- **`ModuleNotFoundError`** - the install did not finish; re-run `pip install -r requirements.txt` and read its output.
- **AI briefing says unavailable** - expected without a key, and not a failure. Everything else is unaffected.

---

## What it does

```
customer history
      │
      ▼
  build profile ......... median payment out, established payees and channels,
      │                   observation window, how far any of it can be trusted
      ▼
  apply rules ........... deterministic; produces the findings, the score
      │                   and the risk level. This is the verdict.
      ▼
  enrich (optional) ..... Gemini: what kind of destination this is, and a
      │                   briefing note explaining the verdict to a human
      ▼
investigation report
```

The two halves are separated on purpose, and the separation runs one way. The
rules reach a verdict first; the model is then handed that verdict and asked to
explain it. **Nothing in the AI layer can raise, lower, create or suppress a
finding.** A model that is missing, slow, rate-limited or wrong changes how a
report reads and never what it says. `POST /api/investigate?enrich=false` returns
the deterministic half alone, and the tests assert both paths agree.

### The rules

| Rule | Fires when | Weight |
|---|---|---|
| `BURST_TO_NEW_PAYEE` | 2+ payments to a newly added payee inside 7 days | 0.40 |
| `UNUSUALLY_LARGE_TRANSFER` | A payment out that is large for this customer *and* for its destination | 0.30 |
| `PATTERN_BREAK` | Recent activity diverges from the customer's own baseline in payees, channels, volatility or pace | 0.20 |
| `ODD_HOURS_ACTIVITY` | 2+ debits between 00:00 and 06:00 | 0.10 |

Rules are held back when the history cannot support them. Under 10 transactions
or under 3 days, nothing is applied and the report says the history is
insufficient rather than that the customer is fine. Under a month of
observation, `PATTERN_BREAK` is withheld — it compares a customer against their
own past, and there is not yet a past to compare to — and the large-transfer
threshold loosens from 2.5x to 3.5x.

### How the findings connect

When more than one rule fires, the report says how the findings relate. Three
findings resting on the same three transfers to the same payee is one episode
seen from three angles, not three separate problems, and reading it the other
way overstates what the customer did. Findings that touch disjoint transactions
are reported as distinct concerns instead. The overlap names which rules share
each transaction.

### What comes back

`risk_level` is one of **ROUTINE**, **INVESTIGATE** or **ESCALATE**. The top band
is named in the field itself rather than left for a caller to work out by
comparing the score against a threshold — three independent rules firing on one
customer is a different instruction to a fraud desk than one. A history too thin
to judge returns ROUTINE with the reason stated and a low confidence, because
"we cannot tell" is not the same claim as "nothing is wrong".

### Grounding

- Every finding carries the identifiers of the transactions it rests on, and
  those identifiers are in `data/sample_customers.json`. In the interface, a
  citation is clickable and shows the source row underneath the finding.
- Destination categories come from `data/risk_typologies.json` and are reported
  with the typology id they matched, so the basis can be read rather than
  trusted. A destination resembling nothing documented is reported as
  unclassified, not pushed into the nearest category.

  Keyword anchors are consulted before embeddings, which is the opposite of how
  this was first built. Measured against the live model, the anchors classified
  every destination they covered correctly while the embeddings placed a
  cryptocurrency exchange under subscriptions — and absolute cosine similarity
  could not separate real destinations (0.547–0.584) from invented company names
  (0.512–0.559) at all, because those ranges overlap. What does separate them is
  the *margin* over the runner-up: something that genuinely belongs to a
  category beats second place clearly, while a meaningless name sits
  near-equidistant from everything. So the anchors carry what a fraud desk
  already knows, embeddings generalise to names nobody listed, and a near-tie is
  reported as no match rather than resolved by guessing. Across 30 destinations
  that yields no false classifications; the failure mode is an honest "could not
  characterise this", which is the safe direction.
- The model's briefing note is checked before it is shown: every transaction id
  and payee it mentions must appear in the findings it was given. A note that
  cites anything else is discarded whole rather than repaired, because a
  briefing referring to a transaction that does not exist reads exactly as
  authoritative as a correct one.

### What it costs to run

A clean history costs nothing at all: with no findings there is nothing to
enrich, so no model is called. A flagged one costs two calls — one batch of
embeddings for the destinations, one generation for the briefing note — on
prompts of roughly 260 to 500 tokens.

Four things keep that down:

- **The typology index is built once and cached to disk.** Committed, so a
  judge's first startup makes no embedding call either.
- **Identical requests are served from memory.** Reviewing the same customer
  five times costs one set of calls, not five — which is what a demo does, and
  what an investigator does returning to a case.
- **Embeddings are cached per destination, not per batch.** The same exchange
  or remittance agent recurs across customers and is embedded once.
- **Destination references are grouped by typology in the prompt.** A pattern
  break citing nine payees across three categories stated each category's prose
  nine times; grouping cut that prompt from 841 tokens to 484.

`GET /health` reports `calls_made` and `cache_hits` so the effect is visible
rather than asserted.

### Checking the live integration

Every AI test breaks the model deliberately — absent key, raising client,
fabricated response, hanging call — because those are the paths that must not
take the report down with them. What tests cannot confirm is that the request
shapes and model name are right, since a wrong model name and an unreachable
network fail identically from inside the application.

```bash
python scripts/verify_gemini.py
```

checks each piece against the live API, lists the model names the key can
actually see if one is wrong, and writes the typology index.

---

## The data

No datasets were provided; everything here was generated. `src/data_generator.py`
builds twenty customers from a fixed seed, so `data/sample_customers.json` can be
regenerated exactly and checked against the code that produced it.

**752 transactions across roughly five months.** Each customer is a persona with
a salary cadence, recurring bills and discretionary spending. Every transaction
carries a date, time, payee, amount, channel, type and description.

| Customer | Outcome | What it exercises |
|---|---|---|
| CUST_003 | INVESTIGATE 0.73 | Three wires to a new cryptocurrency exchange in three days |
| CUST_004 | INVESTIGATE 0.52 | One $12,000 wire abroad against a $95 median |
| CUST_007 | INVESTIGATE 0.45 | Payees and channels turning over wholesale in the closing weeks |
| CUST_008 | INVESTIGATE 0.69 | Repeated transfers to one new payee at 3am |
| CUST_009 | **ESCALATE 0.93** | Three rules at once: large, bursting, overnight |
| CUST_005 | ROUTINE | **A $5,000 investment sweep every month.** Larger than anything else the customer does, and entirely routine |
| CUST_010, CUST_019 | INSUFFICIENT | Too little history to judge — reported as a gap, not a clean bill |
| CUST_020 | ROUTINE | A three-week-old account: real activity, not enough time |
| the rest | ROUTINE | Ordinary people, correctly left alone |

CUST_005 is the case that matters most. Flagging a customer's standing monthly
investment transfer is the same failure as missing CUST_004's wire abroad, and
it is the easier mistake to make.

`data/risk_typologies.json` is the second document the system works over: twelve
categories of destination, what each implies, and — deliberately — what it does
not.

---

## Where this departs from the brief

**Repetition alone is not treated as suspicious.** The specification's edge-case
list suggests that a customer with fifty identical transfers should return
INVESTIGATE on a pattern break. This system returns ROUTINE. Twenty identical
ACH payments to one payee across five months is a standing order — rent, a loan,
a subscription — and flagging it is exactly the over-flagging the problem
statement warns against. A repeated payment becomes interesting when it goes
somewhere new or changes shape, and both of those are already covered.

**Amounts must be positive and dates readable.** Both are refused at the
boundary with a message naming the offending row, rather than carried into the
rules to distort a median or fail somewhere less legible. The API returns 422.

---

## API

| | |
|---|---|
| `GET /health` | Liveness, plus what the AI layer can currently do |
| `GET /api/customers` | The review queue |
| `GET /api/customers/{id}` | Raw history, so any figure in a report can be checked |
| `POST /api/investigate?customer_id=CUST_003` | Review one customer (`&enrich=false` for rules only) |
| `POST /api/investigate/custom` | Review a history supplied in the request body |
| `GET /docs` | OpenAPI |

```bash
curl -X POST "http://localhost:8000/api/investigate?customer_id=CUST_003"
```

---

## Notes on the specification

Three of the specified algorithms could not work as written and were reworked.
They are recorded here because the reasoning is part of the submission.

**`BURST_TO_NEW_PAYEE` could never fire.** An established payee was defined as
one appearing twice or more in the history, and a burst is two or more payments
to one payee — so every burst made its own payee established and excluded
itself. A payee is now new if it is absent from the earlier half of the history,
which is what "newly added" describes.

**`PATTERN_BREAK` could never fire either.** The baseline was the first twenty
transactions and the recent window the last ten; for any history under thirty
entries the baseline swallowed the recent window whole, and nothing could ever
look new. The recent window is now a third of the history with the baseline
being everything before it, and the two never overlap.

**`UNUSUALLY_LARGE_TRANSFER` fired on almost everyone.** Comparing payments
against the median of *all* transactions put the reference between everyday
spending and salary credits, where it described neither; rent and mortgage
payments then flagged on sixteen of the twenty customers. Debits are now
measured against the debit median, and a payment is only unusual if its
destination does not already receive comparable amounts on a recurring basis.
The distinction between recurrence and repetition is doing real work: three
wires to a stranger in three days repeat, but they are the thing being looked
for, whereas rent recurs for months.

The risk-scoring formula was also adjusted. Weights of 0.10 to 0.40 multiplied
by 0.30 could only ever produce 0.33 to 0.42, so every documented band collapsed
onto its floor and a single burst was indistinguishable from a single odd-hours
finding. Weights are now normalised against the heaviest rule, which makes the
formula span the range it was described as spanning.

---

## Built with

Python, FastAPI, Pydantic, httpx, numpy. Gemini for the LLM
(`gemini-3.5-flash-lite`) and embeddings (`gemini-embedding-001`), reached over
the REST endpoint directly rather than through an SDK — it keeps the install
small and does not tie the model name to whichever SDK release knows about it.
No other network calls. The frontend is a single static page with no build step.

## Demo video

*(link to be added)*
