---
title: AR Proactive Context Intelligence
emoji: 🥽
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv RL env for proactive AR glasses AI
tags:
  - openenv
  - reinforcement-learning
  - ar
  - fastapi
  - social-intelligence
---

# AR Proactive Context Intelligence — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://huggingface.co/openenv)
[![Python](https://img.shields.io/badge/python-3.11-brightgreen)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

`ar-proactive-intel` is a fully OpenEnv-compliant reinforcement learning environment that simulates **proactive decision-making for AR glasses** — specifically inspired by the interaction model of **Meta Orion AR glasses**.

### Motivation: Meta Orion & The Proactive AI Problem

Meta Orion places an always-on AI layer over your field of view. The core UX challenge is not *whether* the AI can help — it's *when* it should. An intrusive overlay during a salary negotiation destroys trust. Silence during a cooking emergency is equally bad. This environment formalizes that tradeoff as a learnable RL problem.

> **Unique Selling Point: Silence is a first-class action, rewarded explicitly.**

Unlike most AR assistant benchmarks that only measure helpfulness, this environment rewards the agent for *knowing when not to act* — critical for real-world social acceptance of wearable AI.

---

## Action Space

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `action_type` | string | `assist`, `wait`, `silent` | Primary decision: intervene, gather more info, or stay quiet |
| `assist_type` | string | `navigation`, `productivity`, `social_hint`, `informational`, `none` | Sub-type only relevant when `action_type == assist` |
| `confidence` | float | 0.0 – 1.0 | Agent's self-reported confidence in this decision |

---

## Observation Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `scene_context.location` | string | enum | Where the user is (kitchen, cafe, meeting_room, etc.) |
| `scene_context.activity_signals` | list[str] | — | Observed behavioral signals |
| `scene_context.audio_cues` | list[str] | — | Environmental sounds |
| `scene_context.social_risk` | float | 0.0 – 1.0 | Baseline social sensitivity of the scene |
| `scene_context.time_pressure` | float | 0.0 – 1.0 | Urgency of the situation |
| `scene_context.n_people_nearby` | int | ≥ 0 | Number of bystanders |
| `scene_context.eye_contact_detected` | bool | — | Whether the user is making eye contact with someone |
| `scene_context.user_hands_occupied` | bool | — | Whether user's hands are busy |
| `social_risk` | float | 0.0 – 1.0 | Current dynamic social risk (grows with delay) |
| `context_clarity` | float | 0.0 – 1.0 | Certainty about user intent (grows with delay) |
| `step_number` | int | ≥ 0 | Current step in the episode |
| `interaction_history` | list[str] | — | Log of previous actions and rewards |
| `intent_hint` | string\|null | — | Ground truth hint (easy difficulty only, revealed late) |

---

## Tasks

| Task | Difficulty | Max Steps | Scenario | Correct Strategy |
|------|-----------|-----------|----------|-----------------|
| `easy_cooking` | 🟢 Easy | 5 | Cooking alone in kitchen | Assist quickly with productivity/informational action |
| `medium_cafe` | 🟡 Medium | 8 | Ambiguous cafe intent | Wait for clarity, then assist or continue waiting |
| `hard_social` | 🔴 Hard | 10 | High-stakes social situation | Stay silent — do not interrupt |

### Task Details

**easy_cooking** — The user is alone in their kitchen with clear signals: timer beeping, oven alerts, high time pressure, hands occupied. The agent must identify the intent early and assist with the right type (`productivity` or `informational`). Low social risk means intrusiveness penalties are minimal.

**medium_cafe** — The user is in a social cafe setting with 4–8 people nearby. Signals are ambiguous: they might be working, navigating, or simply relaxing. The optimal strategy is to wait until `context_clarity` rises before committing to an action. Incorrect early assists carry medium intrusiveness penalties.

**hard_social** — The user is in a formal interview, live presentation, or salary negotiation. Social risk is 0.88–0.95. The only correct action is `silent`. Any assist here causes a heavy penalty (`-0.5` intrinsiveness + `-0.4` high-risk assist penalty). The environment explicitly rewards silence: `silence_reward = 0.5` when `social_risk ≥ 0.7`.

---

## Reward Function — Intent Confidence Decay

The reward function models a fundamental tension in proactive AI:

> **The longer you wait, the more you know — but the more awkward your hesitation becomes.**

### How It Works

At every step, two values update dynamically:

- **`context_clarity`** = `min(1.0, 0.3 + step × 0.15)` — increases as agent observes more
- **`social_risk`** = `min(1.0, base_risk + step × 0.12)` — increases as delay grows

This creates a genuine **exploration-exploitation tradeoff**:
- Act too early → wrong intent, wrong assist type, timing penalty
- Act too late → high social risk, intrusiveness penalty even if action is correct
- Stay silent when risk is high → **explicit silence reward**

### Reward Breakdown

| Component | Range | Condition |
|-----------|-------|-----------|
| `intent_accuracy` | 0.0 – 0.4 | Correct action type chosen |
| `assist_relevance` | 0.0 – 0.3 | Correct assist sub-type (assist actions only) |
| `timing_score` | −0.05 – 0.3 | Within optimal step window |
| `silence_reward` | 0.0 – 0.5 | Silent when social_risk ≥ 0.7 |
| `intrusiveness_penalty` | −0.9 – 0.0 | Assisting in high-risk situations |
| **total** | **−1.0 – 1.0** | Clipped sum of all components |

### Episode Grading (0.0 – 1.0)

| Component | Weight | Criteria |
|-----------|--------|----------|
| Correct final action type | 40% | Matches `correct_action` |
| Correct assist sub-type | 20% | Matches `correct_assist_type` (or silent→silent) |
| Timing | 20% | Final step within 1 of `optimal_step` |
| Reward trajectory | 20% | Fraction of positive-reward steps |

---

## Baseline Scores

| Task | Difficulty | Baseline Score (Random Agent) | Baseline Score (Rule-Based LLM) |
|------|-----------|------------------------------|----------------------------------|
| `easy_cooking` | 🟢 Easy | 0.180 | 0.720 |
| `medium_cafe` | 🟡 Medium | 0.150 | 0.510 |
| `hard_social` | 🔴 Hard | 0.120 | 0.840 |
| **Average** | — | **0.150** | **0.690** |

*Rule-based LLM: Qwen2.5-72B-Instruct with social_risk/context_clarity decision rules.*

---

## Local Setup

### Prerequisites

- Python 3.11+
- pip or Docker

### Option 1 — Python (Direct)

```bash
# 1. Navigate to project
cd ar-proactive-intel

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# 5. Verify health
curl http://localhost:7860/health
```

### Option 2 — Docker

```bash
# Build
docker build -t ar-proactive-intel .

# Run
docker run -p 7860:7860 ar-proactive-intel

# Verify
curl http://localhost:7860/health
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET  /health` | GET | Health check |
| `GET  /tasks` | GET | List all available tasks |
| `POST /reset?task_name=easy_cooking` | POST | Reset environment, get first observation |
| `POST /step?task_name=easy_cooking` | POST | Send action, receive reward + next obs |
| `GET  /state?task_name=easy_cooking` | GET | Get agent belief state |
| `POST /grade?task_name=easy_cooking` | POST | Grade current episode (0.0–1.0) |

### Example: Full Episode

```bash
# Reset
curl -X POST "http://localhost:7860/reset?task_name=easy_cooking"

# Step with assist action
curl -X POST "http://localhost:7860/step?task_name=easy_cooking" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assist", "assist_type": "productivity", "confidence": 0.85}'

# Grade the episode
curl -X POST "http://localhost:7860/grade?task_name=easy_cooking"
```

---

## Running Tests

```bash
# From project root
pytest tests/ -v

# Expected output: 9 tests passing
```

---

## Running Inference (LLM Agent)

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860

# Run baseline agent across all 3 tasks
python inference.py
```

Expected output format:
```
[START] task=easy_cooking env=ar-proactive-intel model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=assist reward=1.00 done=true error=null
[END] success=true steps=1 score=0.800 rewards=1.00

[START] task=medium_cafe env=ar-proactive-intel model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=wait reward=0.15 done=false error=null
...
[END] success=true steps=4 score=0.600 rewards=0.15,0.15,0.15,0.50

[SUMMARY] tasks=3 avg_score=0.690 scores=0.800,0.600,0.840
```

---

## HuggingFace Spaces Deployment

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Clone your Space repository
4. Copy all project files into the Space repo
5. Push — the Space will auto-build and expose port 7860

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/ar-proactive-intel
cp -r ar-proactive-intel/* ar-proactive-intel-space/
cd ar-proactive-intel-space
git add .
git commit -m "Initial deploy"
git push
```

The `EXPOSE 7860` directive in the Dockerfile is mandatory for HuggingFace Spaces.

---

## Project Structure

```
ar-proactive-intel/
├── Dockerfile                  # Container definition (port 7860)
├── openenv.yaml                # OpenEnv spec declaration
├── requirements.txt            # Python dependencies
├── inference.py                # Baseline LLM agent runner
├── README.md                   # This file
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI server
│   ├── environment.py          # RL environment core
│   ├── models.py               # Pydantic v2 data models
│   ├── grader.py               # Deterministic episode grader
│   └── reward.py               # Intent Confidence Decay reward engine
├── scenarios/
│   ├── easy/                   # cooking_001–003.json
│   ├── medium/                 # cafe_001–003.json
│   └── hard/                   # interview_001–003.json
└── tests/
    └── test_environment.py     # 9-test pytest suite
```

---

## Why Silence Wins

Standard benchmarks measure task completion. Real AR wearables fail not because the AI can't help, but because **it helps at the wrong moment**. A notification during a job interview costs you the offer. An overlay mid-negotiation costs you salary. This environment trains agents to internalize social context and *choose restraint* — making silence the most sophisticated, highest-value action available.

> Silence is not absence of action. It is the right action, chosen deliberately.

---

## License

MIT License. Built for the Meta AI × HuggingFace OpenEnv Hackathon.
