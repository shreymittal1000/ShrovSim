from simulation.persona.cognition import ReflectComponent
from simulation.persona.common import ChatObservation, PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .reflect_prompts import (
    prompt_insight_and_evidence,
    prompt_memorize_from_conversation,
    prompt_planning_thought_on_conversation,
)


class FishingReflectComponent(ReflectComponent):

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
    ):
        super().__init__(model, model_framework)
        self.prompt_insight_and_evidence = prompt_insight_and_evidence
        self.prompt_planning_thought_on_conversation = (
            prompt_planning_thought_on_conversation
        )
        self.prompt_memorize_from_conversation = prompt_memorize_from_conversation
