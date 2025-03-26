from datetime import datetime

from pathfinder import assistant, system, user
from simulation.persona.cognition.act import ActComponent
from simulation.utils import ModelWandbWrapper

from .vote_prompts import prompt_action_vote, prompt_action_vote_candidate
from .utils import get_universalization_prompt


class VotingActComponent(ActComponent):

    def __init__(
        self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg
    ):
        super().__init__(model, model_framework, cfg)

    def choose_vote(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
    ):
        res, html = prompt_action_vote(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            self.cfg
        )
        res = int(res)
        return res, [html]


class VotingActComponentCandidate(ActComponent):
    
    def __init__(
        self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg
    ):
        super().__init__(model, model_framework, cfg)

    def choose_vote(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
    ):
        res, html = prompt_action_vote_candidate(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            self.cfg
        )
        res = int(res)
        return res, [html]