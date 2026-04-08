from typing import List
from app.models import Action, ScenarioDefinition


def grade_episode(
    actions: List[Action],
    scenario: ScenarioDefinition,
    rewards: List[float],
    max_steps: int,
) -> float:
    """
    Deterministic grader that scores 0.0 - 1.0 for a full episode.

    Scoring breakdown:
    - Correct final action type: 40 points
    - Correct assist type (if applicable): 20 points
    - Timing within 1 step of optimal: 20 points
    - Reward trajectory quality: 20 points
    """
    if not actions:
        return 0.0001

    score = 0.0
    final_action = actions[-1]

    # Correct action type (40%)
    if final_action.action_type == scenario.correct_action:
        score += 0.4

    # Correct assist type (20%)
    if (
        final_action.action_type == "assist"
        and final_action.assist_type == scenario.correct_assist_type
    ):
        score += 0.2
    elif scenario.correct_action == "silent" and final_action.action_type == "silent":
        score += 0.2

    # Timing score (20%)
    final_step = len(actions)
    if scenario.correct_action != "silent":
        step_diff = abs(final_step - scenario.optimal_step)
        if step_diff == 0:
            score += 0.2
        elif step_diff == 1:
            score += 0.1
    else:
        score += 0.2

    # Reward trajectory (20%)
    if rewards:
        positive_rewards = sum(1 for r in rewards if r > 0)
        trajectory_score = positive_rewards / len(rewards)
        score += 0.2 * trajectory_score

    return round(min(0.9999, max(0.0001, score)), 4)
