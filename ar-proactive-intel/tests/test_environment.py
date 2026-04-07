import pytest
from app.environment import ARProactiveEnvironment
from app.models import Action


def test_reset_returns_observation():
    env = ARProactiveEnvironment(task_name="easy_cooking")
    result = env.reset()
    assert result.observation is not None
    assert result.observation.step_number == 0
    assert 0.0 <= result.observation.social_risk <= 1.0
    assert 0.0 <= result.observation.context_clarity <= 1.0


def test_step_returns_reward():
    env = ARProactiveEnvironment(task_name="easy_cooking")
    env.reset()
    action = Action(action_type="assist", assist_type="productivity", confidence=0.8)
    result = env.step(action)
    assert isinstance(result.reward, float)
    assert result.observation is not None
    assert isinstance(result.done, bool)


def test_state_returns_belief():
    env = ARProactiveEnvironment(task_name="easy_cooking")
    env.reset()
    state = env.state()
    assert len(state.intent_probabilities) > 0
    assert 0.0 <= state.social_risk_level <= 1.0


def test_hard_social_rewards_silence():
    env = ARProactiveEnvironment(task_name="hard_social")
    env.reset()
    action = Action(action_type="silent", assist_type="none", confidence=0.9)
    result = env.step(action)
    assert result.reward > 0.0


def test_episode_ends_at_max_steps():
    env = ARProactiveEnvironment(task_name="easy_cooking")
    env.reset()
    for _ in range(env.max_steps):
        action = Action(action_type="wait", assist_type="none", confidence=0.5)
        result = env.step(action)
    assert result.done is True


def test_done_episode_raises_on_step():
    env = ARProactiveEnvironment(task_name="easy_cooking")
    env.reset()
    for _ in range(env.max_steps):
        env.step(Action(action_type="wait", assist_type="none", confidence=0.5))
    with pytest.raises(RuntimeError):
        env.step(Action(action_type="assist", assist_type="none", confidence=0.5))


def test_reward_in_valid_range():
    env = ARProactiveEnvironment(task_name="medium_cafe")
    env.reset()
    for action_type in ["assist", "wait", "silent"]:
        env2 = ARProactiveEnvironment(task_name="medium_cafe")
        env2.reset()
        result = env2.step(
            Action(action_type=action_type, assist_type="none", confidence=0.5)
        )
        assert -1.0 <= result.reward <= 1.0, f"Reward out of range for {action_type}"


def test_grader_returns_valid_score():
    from app.grader import grade_episode

    env = ARProactiveEnvironment(task_name="easy_cooking")
    env.reset()
    env.step(Action(action_type="assist", assist_type="productivity", confidence=0.9))
    score = grade_episode(
        actions=env.get_actions(),
        scenario=env._scenario,
        rewards=env.get_rewards(),
        max_steps=env.max_steps,
    )
    assert 0.0 <= score <= 1.0


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        ARProactiveEnvironment(task_name="nonexistent_task")
