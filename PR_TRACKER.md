# 📊 PR Tracker

**Purpose:** Central hub for tracking all active PRs, their blockers, dependencies, and what needs to happen next.  
**Updated:** After every PR creation/update.  
**Read First:** When planning next session.

---

## Active PRs (Draft Status)

### PR #4: "24/7 crypto spot trading with always-on ETP short strategy"

**Status:** 🔴 DRAFT - BLOCKED  
**Priority:** 🔴 URGENT (Foundation PR - all others depend on it)  
**Created:** ~3 days ago  
**Owner:** Copilot (with NedFerg oversight)

**What This PR Does:**
- Adds comprehensive bull/bear ETP (short ETF) strategy
- Implements 4-state phase machine (accumulation → bull_alt_season → alt_cascade → bear_market)
- Adds phase transition signal detection (BTC breakout, alt momentum, market top)
- Adds `phase_transitions`, `rotations`, `signal_confidence` SQLite tables
- Enables allocation-aware trading (never > 30% ETF allocation)
- Designed for always-on 24/7 spot trading during bull/bear cycles

**Current Blockers:**
1. ❌ `requirements.txt` missing (CRITICAL)
   - Fix: Create with essential deps only
   - Impact: Bot won't install on Replit/DigitalOcean  

2. ❌ `pyproject.toml` bloat (~1000+ PyTorch packages)
   - Fix: Strip to ~5 essential packages
   - Impact: Installation fails, breaks CI/CD  

3. ❌ Torch import not wrapped (rl_agent.py line 8)
   - Fix: Wrap in try/except, lazy-load
   - Impact: Bot fails if PyTorch not installed  

4. ❌ No end-to-end test
   - Fix: Run paper mode, verify startup + signal generation
   - Impact: Unclear if bot runs at all

**Merge Dependencies:** None (this is foundation)

**What Needs to Happen Next:**
1. [ ] Clean up requirements.txt + pyproject.toml
2. [ ] Verify torch lazy-loading works
3. [ ] Test bot startup in paper mode
4. [ ] Verify MA strategy executes without errors
5. [ ] Then implement phase machine logic
6. [ ] Add SQLite schema for phase_transitions table
7. [ ] Test phase detection on historical data
8. [ ] Ready for review

**Files Involved:**
- `pyproject.toml` (needs cleanup)
- `requirements.txt` (needs creation)
- `project/agents/rl_agent.py` (needs torch wrapping)
- `project/main.py` (phase machine entry point)
- `project/brokers/etf_hedging.py` (ETF allocation logic)
- `project/trades.db` (schema extension)

**Estimated Effort:** 2-3 sessions (after dependencies fixed)

---

### PR #5: "Harden sandbox: resilient price feed, retry backoff, threshold-based kill switch, RL agent validation"

**Status:** 🟡 DRAFT - BLOCKED  
**Priority:** 🟡 HIGH  
**Created:** 3 hours ago  
**Owner:** Copilot (with NedFerg oversight)

**What This PR Does:**
- Adds resilient price feed (retry backoff for API failures)
- Adds threshold-based kill switch (stops trading if equity drops X%)
- Adds RL agent validation (ensures checkpoint valid before trading)
- Improves sandbox stability for live price testing

**Blocker:** 
- ❌ Depends on PR #4 (needs phase machine foundation)

**Merge Order:** PR #4 → PR #5

**What Needs to Happen Next:**
1. [ ] Merge PR #4 first
2. [ ] Then implement price feed resilience
3. [ ] Add kill switch with equity threshold
4. [ ] Add RL validation checks
5. [ ] Test in sandbox mode with live prices

**Estimated Effort:** 1-2 sessions (after PR #4 done)

---

### PR #6: "Add ETF hedging layer (ETHD/SETH) to replace disabled futures for US users"

**Status:** 🟡 DRAFT - BLOCKED  
**Priority:** 🟡 HIGH  
**Created:** Yesterday  
**Owner:** Copilot (with NedFerg oversight)

**What This PR Does:**
- Replaces disabled futures trading with short ETF hedging
- Adds ETHD (Ethereum 2x short) + SETH (Ethereum 1x short) support
- Implements smart allocation (max 30% per direction)
- Enables downside capture during bear phases

**Blocker:**
- ❌ Depends on PR #5 (needs sandbox hardening first)

**Merge Order:** PR #4 → PR #5 → PR #6

**Note:** ETF layer already partially exists in code; this PR completes + hardens it

**What Needs to Happen Next:**
1. [ ] Merge PR #4, then PR #5
2. [ ] Complete ETF allocation logic
3. [ ] Test 30% cap enforcement
4. [ ] Verify long/short mode switching
5. [ ] Test leverage multiplier (2x for longs, -2x for shorts)

**Estimated Effort:** 1 session (after PR #5 done)

---

### PR #10: "Fix backtesting pipeline: synthetic OHLCV fallback when Kraken API unavailable in CI"

**Status:** 🟡 DRAFT - BLOCKED  
**Priority:** 🟡 MEDIUM  
**Created:** Yesterday  
**Owner:** Copilot (with NedFerg oversight)

**What This PR Does:**
- Adds synthetic OHLCV data generation when API unavailable
- Fixes CI/CD pipeline (firewall blocks Kraken API in GitHub Actions)
- Enables historical backtesting without external API calls

**Blocker:**
- ❌ CI/CD firewall (GitHub Actions can't reach Kraken)

**Workaround:**
- Use synthetic data fallback
- Fetch historical data locally, commit to repo

**Merge Order:** Can merge independently after #4-#6 stabilize

**What Needs to Happen Next:**
1. [ ] Implement synthetic OHLCV generator
2. [ ] Add fallback logic to data fetch
3. [ ] Test CI pipeline with synthetic data
4. [ ] Verify backtesting passes in CI

**Estimated Effort:** 1 session (lower priority)

---

## Merge Order (Locked)

```
PR #4 (Foundation)
  ↓ (all tests pass)
PR #5 (Sandbox Hardening)
  ↓ (all tests pass)
PR #6 (ETF Hedging)
  ↓ (all tests pass)
PR #10 (CI/Backtesting) [can happen in parallel]
  ↓
MERGE TO MASTER
```

**Do not merge any PR out of order.**

---

## Current Bottleneck

**PR #4 is blocked on:**
1. requirements.txt cleanup
2. pyproject.toml bloat removal
3. Torch lazy-loading verification
4. End-to-end test (paper mode)

**Unblock PR #4 = Unblock everything**

**Priority 1 for next session: Fix these 3 dependencies.**

---

## Status Symbols

| Symbol | Meaning |
|--------|---------|
| 🟢 | Ready to merge |
| 🟡 | In progress / waiting on dependency |
| 🔴 | Blocked / urgent |
| ⏸️ | Paused |

---

## How to Update

After each PR update:
1. Change status symbol
2. Add blocker if new one discovered
3. Update "What Needs to Happen Next"
4. Estimate new effort if changed
5. Check merge order still valid