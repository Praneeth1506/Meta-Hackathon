from app.models import Action, ScenarioDefinition


def compute_context_clarity(step: int, base_clarity: float = 0.3) -> float:
    """Context clarity increases as agent observes more steps."""
    return min(1.0, base_clarity + step * 0.15)


def compute_social_risk(step: int, base_risk: float = 0.1) -> float:
    """Social risk increases as agent delays — standing with glasses doing nothing is awkward."""
    return min(1.0, base_risk + step * 0.12)


def compute_reward(
    action: Action,
    step: int,
    scenario: ScenarioDefinition,
    base_social_risk: float,
) -> dict:
    """
    Core reward function with Intent Confidence Decay.

    The longer the agent waits:
    - context_clarity increases (more info available)
    - social_risk increases (delay itself becomes intrusive)

    This creates a genuine exploration-exploitation tradeoff.
    """
    context_clarity = compute_context_clarity(step, 0.3)
    social_risk = compute_social_risk(step, base_social_risk)

    intent_accuracy = 0.0
    assist_relevance = 0.0
    timing_score = 0.0
    silence_reward = 0.0
    intrusiveness_penalty = 0.0

    # --- Silence decision ---
    if action.action_type == "silent":
        if social_risk >= 0.7:
            silence_reward = 0.5
            timing_score = 0.3
        elif social_risk >= 0.5:
            silence_reward = 0.2
        else:
            if scenario.correct_action == "silent":
                silence_reward = 0.4
            else:
                intrusiveness_penalty = -0.2

    # --- Assist decision ---
    elif action.action_type == "assist":
        if scenario.correct_action == "assist":
            intent_accuracy = 0.4
            if action.assist_type == scenario.correct_assist_type:
                assist_relevance = 0.3
            if abs(step - scenario.optimal_step) <= 1:
                timing_score = 0.3
            elif abs(step - scenario.optimal_step) <= 2:
                timing_score = 0.15
        else:
            intrusiveness_penalty = -0.5 if social_risk >= 0.7 else -0.3

        if social_risk >= 0.7 and action.action_type == "assist":
            intrusiveness_penalty -= 0.4

    # --- Wait decision ---
    elif action.action_type == "wait":
        if context_clarity < 0.5:
            timing_score = 0.15
        elif scenario.correct_action == "wait":
            timing_score = 0.2
            intent_accuracy = 0.1
        else:
            timing_score = -0.05

    total = (
        intent_accuracy
        + assist_relevance
        + timing_score
        + silence_reward
        + intrusiveness_penalty
    )
    total = max(-1.0, min(1.0, total))

    return {
        "total": total,
        "intent_accuracy": intent_accuracy,
        "assist_relevance": assist_relevance,
        "timing_score": timing_score,
        "silence_reward": silence_reward,
        "intrusiveness_penalty": intrusiveness_penalty,
        "context_clarity": context_clarity,
        "social_risk": social_risk,
    }
