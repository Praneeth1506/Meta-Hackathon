from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn

from app.environment import ARProactiveEnvironment
from app.models import Action
from app.grader import grade_episode

app = FastAPI(
    title="AR Proactive Context Intelligence",
    description="OpenEnv environment for proactive AR decision-making",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instances per task
_envs: Dict[str, ARProactiveEnvironment] = {}


def get_env(task_name: str = "easy_cooking") -> ARProactiveEnvironment:
    if task_name not in _envs:
        _envs[task_name] = ARProactiveEnvironment(task_name=task_name)
    return _envs[task_name]


@app.post("/reset")
async def reset(task_name: str = "easy_cooking") -> dict:
    env = get_env(task_name)
    result = env.reset()
    return result.model_dump()


@app.post("/step")
async def step(action: Action, task_name: str = "easy_cooking") -> dict:
    env = get_env(task_name)
    try:
        result = env.step(action)
    except AssertionError:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first before calling /step.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


@app.get("/state")
async def state(task_name: str = "easy_cooking") -> dict:
    env = get_env(task_name)
    return env.state().model_dump()


@app.post("/grade")
async def grade(task_name: str = "easy_cooking") -> dict:
    env = get_env(task_name)
    score = grade_episode(
        actions=env.get_actions(),
        scenario=env._scenario,
        rewards=env.get_rewards(),
        max_steps=env.max_steps,
    )
    return {"score": score, "task": task_name}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks() -> dict:
    return {
        "tasks": [
            {"name": "easy_cooking", "difficulty": "easy", "max_steps": 5},
            {"name": "medium_cafe", "difficulty": "medium", "max_steps": 8},
            {"name": "hard_social", "difficulty": "hard", "max_steps": 10},
        ]
    }


def start() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    start()
