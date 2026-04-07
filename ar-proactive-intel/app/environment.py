import json
import random
from pathlib import Path
from typing import List, Optional

from app.models import (
    Action,
    AgentBeliefState,
    Observation,
    ResetResult,
    ScenarioDefinition,
    StepResult,
)
from app.reward import compute_reward, compute_context_clarity, compute_social_risk


SCENARIOS_PATH = Path(__file__).parent.parent / "scenarios"

TASK_TO_DIR = {
    "easy_cooking": "easy",
    "medium_cafe": "medium",
    "hard_social": "hard",
}

MAX_STEPS = {
    "easy_cooking": 5,
    "medium_cafe": 8,
    "hard_social": 10,
}


class ARProactiveEnvironment:
    def __init__(self, task_name: str = "easy_cooking"):
        if task_name not in TASK_TO_DIR:
            raise ValueError(
                f"Unknown task: {task_name}. Choose from {list(TASK_TO_DIR.keys())}"
            )
        self.task_name = task_name
        self.max_steps = MAX_STEPS[task_name]
        self._scenario: Optional[ScenarioDefinition] = None
        self._step_count: int = 0
        self._done: bool = False
        self._actions: List[Action] = []
        self._rewards: List[float] = []
        self._history: List[str] = []

    def _load_random_scenario(self) -> ScenarioDefinition:
        scenario_dir = SCENARIOS_PATH / TASK_TO_DIR[self.task_name]
        files = list(scenario_dir.glob("*.json"))
        chosen = random.choice(files)
        with open(chosen) as f:
            data = json.load(f)
        return ScenarioDefinition(**data)

    def _build_observation(self) -> Observation:
        assert self._scenario is not None
        step = self._step_count
        context_clarity = compute_context_clarity(step, 0.3)
        social_risk = compute_social_risk(step, self._scenario.social_risk)

        intent_hint = None
        if context_clarity > 0.7 and self._scenario.difficulty == "easy":
            intent_hint = self._scenario.ground_truth_intent

        return Observation(
            scene_context=self._scenario.scene,
            audio_cues=self._scenario.scene.audio_cues,
            social_risk=min(1.0, social_risk),
            context_clarity=context_clarity,
            step_number=self._step_count,
            interaction_history=self._history.copy(),
            intent_hint=intent_hint,
        )

    def reset(self) -> ResetResult:
        self._scenario = self._load_random_scenario()
        self._step_count = 0
        self._done = False
        self._actions = []
        self._rewards = []
        self._history = []
        obs = self._build_observation()
        return ResetResult(
            observation=obs,
            info={
                "task": self.task_name,
                "scenario_id": self._scenario.scenario_id,
                "difficulty": self._scenario.difficulty,
                "max_steps": self.max_steps,
            },
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        assert self._scenario is not None

        self._step_count += 1
        self._actions.append(action)

        reward_breakdown = compute_reward(
            action=action,
            step=self._step_count,
            scenario=self._scenario,
            base_social_risk=self._scenario.social_risk,
        )
        reward = reward_breakdown["total"]
        self._rewards.append(reward)

        self._history.append(
            f"step={self._step_count} action={action.action_type} "
            f"assist_type={action.assist_type} reward={reward:.2f}"
        )

        done = False
        if self._step_count >= self.max_steps:
            done = True
        if action.action_type == "assist" and self._scenario.correct_action == "assist":
            if self._step_count >= self._scenario.optimal_step:
                done = True
        if (
            action.action_type == "silent"
            and self._scenario.correct_action == "silent"
        ):
            done = True

        self._done = done
        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "reward_breakdown": reward_breakdown,
                "scenario_id": self._scenario.scenario_id,
                "step": self._step_count,
                "actions_taken": len(self._actions),
            },
        )

    def state(self) -> AgentBeliefState:
        assert self._scenario is not None
        step = self._step_count
        context_clarity = compute_context_clarity(step, 0.3)
        social_risk = compute_social_risk(step, self._scenario.social_risk)

        intent_probs = {
            intent: round(1.0 / len(self._scenario.intents), 3)
            for intent in self._scenario.intents
        }
        if context_clarity > 0.6:
            gt = self._scenario.ground_truth_intent
            if gt in intent_probs:
                intent_probs[gt] = min(1.0, intent_probs[gt] + context_clarity * 0.4)

        return AgentBeliefState(
            intent_probabilities=intent_probs,
            current_step=self._step_count,
            social_risk_level=min(1.0, social_risk),
            context_clarity=context_clarity,
            interaction_history=self._history.copy(),
            previous_actions=[a.action_type for a in self._actions],
            episode_done=self._done,
        )

    def get_actions(self) -> List[Action]:
        return self._actions.copy()

    def get_rewards(self) -> List[float]:
        return self._rewards.copy()
