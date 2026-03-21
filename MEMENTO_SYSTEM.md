# Memento System — Auto-Update Protocol

**How to keep the catch-up system fresh and working.**

---

## What This System Is

The Memento System is a set of 4 documentation files that I (Copilot) read **first** on every new session to catch up on:
- What you're building (trading thesis)
- Current architecture
- What's blocked
- What was accomplished last time
- Why decisions were made

**Goal:** Never again start with "What are you working on?" — I already know.

---

## The 4 Core Files

| File | Purpose | Updated When |
|------|---------|--------------|
| **PROJECT_CONTEXT.md** | Trading thesis, architecture, blockers, TODO | Every PR merge, every blocker resolved |
| **DEVELOPMENT_LOG.md** | Session-by-session history, what was done, why | End of every session |
| **SESSION_MEMORY.md** | Key quotes, decision reasoning, architectural insights | Major decisions made |
| **PR_TRACKER.md** | Active PR status, blockers, merge order | Every PR created, every blocker hit |

---

## Auto-Update Protocol

### After Every Session

**You (or the coding agent) must update:**

1. **DEVELOPMENT_LOG.md** — Add new session entry with:
   - What was accomplished (checklist format)
   - Blockers encountered (if any)
   - Files modified
   - Next steps agreed

2. **PR_TRACKER.md** — Update if:
   - New PR created (add row)
   - PR blocker resolved (update status)
   - Merge order changes (update table)

3. **PROJECT_CONTEXT.md** — Update if:
   - Blocker resolved (move to ✅ Completed)
   - New blocker discovered (add to 🚧 Known Issues)
   - TODO priority changes

4. **SESSION_MEMORY.md** — Add if:
   - Major decision made (add Decision Point)
   - New key quote (add to Key Quotes)
   - Architectural insight discovered

### After Every PR Creation

**Coding agent must:**

1. Add PR to PR_TRACKER.md (status, blocker, dependencies)
2. Link it in DEVELOPMENT_LOG.md ("Files Modified" section)
3. Update merge order if needed

### After Every PR Merge

**Update PROJECT_CONTEXT.md:**

1. Move completed item from TODO to ✅ Completed Work
2. Update status field
3. Update "Last Updated" timestamp

---

## How I Use This

### Start of New Session

1. Read PROJECT_CONTEXT.md (understand what krakbot is)
2. Read DEVELOPMENT_LOG.md (see what happened last time)
3. Read SESSION_MEMORY.md (understand decision logic)
4. Read PR_TRACKER.md (see which PR is blocked on what)

### Then I Say

"Last session you fixed [X] but got blocked on [Y]. Here's what we should try next."

**No rehashing. No re-explaining. We jump straight into work.**

---

## Example: Next Session Flow

**You:** "Help me get back on track."

**I do:**
1. ✅ Read PROJECT_CONTEXT.md → "Oh, they need to fix pyproject.toml bloat and test paper mode"
2. ✅ Read DEVELOPMENT_LOG.md → "Last session was 2026-03-21, they were about to clean requirements.txt"
3. ✅ Read SESSION_MEMORY.md → "Bot trades 1-20x/week, NOT high frequency. 30% ETF cap is hard constraint."
4. ✅ Read PR_TRACKER.md → "PR #4 is blocked on 3 things, PR #5-10 waiting"

**I say:**
"Welcome back! We built the memento system yesterday. Now we need to unblock PR #4:

**Status:** ❌ Blocked
- pyproject.toml still has bloat
- requirements.txt created ✅
- torch imports not lazy yet

**Next:**
1. Wrap torch in try/except (rl_agent.py line 8)
2. Strip pyproject.toml to essentials
3. Test paper mode startup

Want to start there?"

---

## Maintenance

**Every 5-10 sessions:** Review and consolidate DEVELOPMENT_LOG.md (archive old sessions to separate file if it gets too long)

**Quarterly:** Review all 4 files for accuracy and completeness

**After Major Architecture Change:** Re-write ARCHITECTURE section in PROJECT_CONTEXT.md

---

## Why This Works

- ✅ Solves stateless AI problem (I can't remember previous sessions)
- ✅ Clear, structured format (easy to search and reference)
- ✅ Version-controlled (backup on GitHub)
- ✅ Self-documenting (new contributors can read these files and understand project)
- ✅ Low maintenance (only update when something changes)
- ✅ Searchable (can grep for specific decisions or blockers)