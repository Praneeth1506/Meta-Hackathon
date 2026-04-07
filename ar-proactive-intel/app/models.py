from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from enum import Enum


class Location(str, Enum):
    HOME = "home"
    CAFE = "cafe"
    OFFICE = "office"
    STREET = "street"
    SOCIAL_EVENT = "social_event"
    KITCHEN = "kitchen"
    MEETING_ROOM = "meeting_room"


class SceneContext(BaseModel):
    location: Location
    activity_signals: List[str]
    audio_cues: List[str]
    social_risk: float = Field(ge=0.0, le=1.0)
    time_pressure: float = Field(ge=0.0, le=1.0)
    n_people_nearby: int = Field(ge=0)
    eye_contact_detected: bool
    user_hands_occupied: bool


class Observation(BaseModel):
    scene_context: SceneContext
    audio_cues: List[str]
    social_risk: float = Field(ge=0.0, le=1.0)
    context_clarity: float = Field(ge=0.0, le=1.0)
    step_number: int
    interaction_history: List[str] = Field(default_factory=list)
    intent_hint: Optional[str] = None


class Action(BaseModel):
    action_type: Literal["assist", "wait", "silent"]
    assist_type: Literal[
        "navigation", "productivity", "social_hint", "informational", "none"
    ] = "none"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class Reward(BaseModel):
    total: float
    intent_accuracy: float
    assist_relevance: float
    timing_score: float
    silence_reward: float
    intrusiveness_penalty: float
    breakdown: Dict[str, float]


class AgentBeliefState(BaseModel):
    intent_probabilities: Dict[str, float]
    current_step: int
    social_risk_level: float
    context_clarity: float
    interaction_history: List[str]
    previous_actions: List[str]
    episode_done: bool


class ScenarioDefinition(BaseModel):
    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"]
    task_name: str
    scene: SceneContext
    ground_truth_intent: str
    correct_action: Literal["assist", "wait", "silent"]
    correct_assist_type: str
    optimal_step: int
    social_risk: float
    description: str
    intents: List[str]


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: Observation
    info: Dict[str, Any]
