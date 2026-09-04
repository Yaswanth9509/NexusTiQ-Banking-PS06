TRACK_ID=PS06

# Banking Risk Investigation Assistant

**Transaction Risk Investigation System - NexusTiQ 24**

## Problem Statement

Build an investigation assistant for a bank's fraud desk that reviews customer transaction histories against risk rules to identify suspicious activity patterns.

The system analyzes months of transaction history and checks for:
- **Unusually large transfers** (>2.5x customer average)
- **Bursts to new payees** (multiple transfers to new recipient in short time)
- **Odd-hours activity** (transactions at unusual times)
- **Pattern breaks** (activity inconsistent with customer's history)

Every finding cites specific transactions from the input history. **The system never concludes fraud** - it flags, explains, and escalates to human investigators.

---

## Project Structure

```
banking-risk-investigation/
├── app.py                          # FastAPI application
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── .env.example & .gitignore
│
├── src/
│   ├── __init__.py
│   ├── models.py                   # Pydantic data models
│   ├── analyzer.py                 # TransactionAnalyzer coordinator
│   ├── utils.py                    # Helper functions
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── base.py                 # RiskRule base class
│   │   ├── large_transfer.py       # UnusuallyLargeTransferRule
│   │   ├── new_payee_burst.py      # BurstToNewPayeeRule
│   │   ├── odd_hours.py            # OddHoursActivityRule
│   │   └── pattern_break.py        # PatternBreakRule
│   └── data_generator.py           # Sample data generation
│
├── data/
│   └── sample_customers.json       # 20 sample customer histories
│
└── tests/
    ├── test_analyzer.py
    ├── test_rules.py
    └── test_edge_cases.py
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
python app.py
```

Application starts on `http://localhost:8000`

### 3. Test Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### List Customers
```bash
curl http://localhost:8000/api/customers
```

#### Investigate Customer
```bash
curl -X POST "http://localhost:8000/api/investigate?customer_id=CUST_001"
```

#### Interactive API Docs
Visit: `http://localhost:8000/docs`

---

## Risk Rules

| Rule | Detects | Threshold | Weight |
|------|---------|-----------|--------|
| UNUSUALLY_LARGE_TRANSFER | Transfers >2.5x average | 2.5x median | 0.30 |
| BURST_TO_NEW_PAYEE | Multiple transfers to new recipient | 2+ in 7 days | 0.40 |
| ODD_HOURS_ACTIVITY | Transactions 00:00-05:59 AM | 2+ txns | 0.10 |
| PATTERN_BREAK | Activity diverges from baseline | 4 checks | 0.20 |

---

## Sample Data

20 customer test cases covering:
- **5 Normal/Routine cases** - ROUTINE classification
- **5 Suspicious cases** - INVESTIGATE classification
- **5 Edge cases** - Context-dependent assessment
- **5 Mixed cases** - Various patterns

Each customer has 10-17 transactions over 30-50 days.

---

## Risk Scoring

| Findings | Risk Score | Risk Level | Confidence |
|----------|-----------|-----------|-----------|
| 0 | 0.0 | ROUTINE | 0.90 |
| 1 | 0.4-0.6 | INVESTIGATE | 0.70 |
| 2 | 0.65-0.75 | INVESTIGATE | 0.70-0.80 |
| 3+ | 0.80-1.0 | ESCALATE | 0.95 |

---

## Profile Maturity Levels

The system adjusts rule sensitivity based on data availability:

**INSUFFICIENT** (< 10 txns OR < 3 days):
- Do not apply rules
- Return: "Insufficient data for analysis"
- risk_score = 0.0

**EMERGING** (10-29 txns AND 3-29 days):
- Apply rules loosely
- Large transfer threshold: 3.5x median
- Skip PATTERN_BREAK rule

**ESTABLISHED** (30+ txns AND 30+ days):
- Apply all rules strictly
- Large transfer threshold: 2.5x median
- Full pattern analysis enabled

---

## Key Features

✅ **No false positives** - System doesn't over-flag. Routine activity stays routine.  
✅ **Traceable findings** - Every finding cites specific transactions with IDs and amounts  
✅ **Context-aware** - Builds customer profile before checking rules  
✅ **Graceful escalation** - Uncertain cases escalated rather than guessed  
✅ **Production-ready** - Proper error handling, validation, logging  

---

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_analyzer.py
```

Run with coverage:
```bash
pytest --cov=src tests/
```

---

## Technology Stack

- **Framework**: FastAPI
- **Data Validation**: Pydantic
- **API Server**: Uvicorn
- **Testing**: Pytest
- **Python**: 3.8+

---

## Development Timeline

- **Hour 1-4**: Setup + core analyzer
- **Hour 4-8**: Risk rules implementation
- **Hour 8-12**: API + data generation
- **Hour 12-16**: Testing + edge cases
- **Hour 16-24**: Polish + demo + submission

---

## Architecture

```
CustomerHistory (JSON)
    ↓
TransactionAnalyzer
    ├─ Build Profile (establish baseline)
    ├─ Check Rules (apply 4 risk rules)
    ├─ Collect Findings
    └─ Generate Report
    ↓
InvestigationReport (JSON)
    └─ risk_level + findings + recommendations
```

---

**Built for NexusTiQ 24 Hackathon**
