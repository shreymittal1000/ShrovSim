import typing

from pathfinder import assistant, system, user
from simulation.utils import ModelWandbWrapper

from .component import Component


class PlanComponent(Component):
    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
    ):
        super().__init__(model, model_framework)

    def chat_react(self):
        # There are 2 persona, the first is initiator, the second is responder.
        # Converse.generate_convo
        # Converse.generate_convo_summary
        pass

    def revise_self_indentity(self):
        # NOTE: future. Update persona's self identity, given new experience.
        pass

    def should_react(self):
        # NOTE: future
        pass

    def wait_react(self):
        # NOTE: future
        pass

    def create_react(self):
        # NOTE: future
        pass
