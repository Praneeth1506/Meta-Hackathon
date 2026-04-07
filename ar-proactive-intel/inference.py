"""
inference.py — AR Proactive Context Intelligence Baseline Agent
Mandatory stdout format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file if present (does nothing if .env doesn't exist)
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "ar-proactive-intel"
MAX_STEPS_PER_TASK = 10
SUCCESS_THRESHOLD = 0.5

TASKS = ["easy_cooking", "medium_cafe", "hard_social"]

SYSTEM_PROMPT = """
You are an AI agent controlling AR glasses that must decide when and how to assist a user.

At each step you receive a scene context including:
- location: where the user is
- activity_signals: what the user is doing
- audio_cues: what sounds are present
- social_risk: float 0.0-1.0 (higher = more socially sensitive)
- context_clarity: float 0.0-1.0 (higher = more certain about intent)

You must respond with a JSON object exactly like this:
{
  "action_type": "assist" | "wait" | "silent",
  "assist_type": "navigation" | "productivity" | "social_hint" | "informational" | "none",
  "confidence": 0.0 to 1.0
}

Rules:
- If social_risk > 0.7, strongly prefer "silent"
- If context_clarity < 0.4, prefer "wait"
- If context_clarity > 0.6 and social_risk < 0.3, prefer "assist"
- Always return valid JSON only, no explanation
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def call_env(endpoint: str, method: str = "POST", data: dict = None) -> dict:
    url = f"{ENV_URL}/{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=data or {}, timeout=30)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_agent_action(client: OpenAI, observation: dict, step: int) -> dict:
    obs_str = json.dumps(observation, indent=2)
    user_prompt = f"Step {step}. Current observation:\n{obs_str}\n\nDecide your action."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"action_type": "wait", "assist_type": "none", "confidence": 0.5}


def run_task(client: OpenAI, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    reset_result = call_env(f"reset?task_name={task_name}")
    if "error" in reset_result:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    observation = reset_result.get("observation", {})
    rewards: List[float] = []
    steps_taken = 0
    done = False
    score = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if done:
                break

            action_dict = get_agent_action(client, observation, step)
            action_str = action_dict.get("action_type", "wait")

            step_result = call_env(
                f"step?task_name={task_name}",
                method="POST",
                data=action_dict,
            )

            error = step_result.get("error", None)
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            observation = step_result.get("observation", observation)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

        grade_result = call_env(f"grade?task_name={task_name}")
        score = float(grade_result.get("score", 0.0))
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    all_scores = []
    for task in TASKS:
        score = run_task(client, task)
        all_scores.append(score)
    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"\n[SUMMARY] tasks={len(TASKS)} avg_score={avg:.3f} "
        f"scores={','.join(f'{s:.3f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
