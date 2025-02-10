from typing import Any

from simulation.persona import (
    ActComponent,
    ConverseComponent,
    PerceiveComponent,
    PersonaAgent,
    PersonaOberservation,
    PlanComponent,
    ReflectComponent,
    RetrieveComponent,
    StoreComponent,
)
from simulation.persona.common import (
    ChatObservation,
    PersonaAction,
    PersonaActionChat,
    PersonaActionHarvesting,
    PersonaIdentity,
)
from simulation.persona.embedding_model import EmbeddingModel
from simulation.persona.memory import AssociativeMemory, Scratch
from simulation.scenarios.common.environment import HarvestingObs
from simulation.utils import ModelWandbWrapper

from .cognition import (
    FishingActComponent,
    FishingConverseComponent,
    FishingPlanComponent,
    FishingReflectComponent,
    FishingStoreComponent,
)


class FishingPersona(PersonaAgent):
    last_collected_resource_num: int
    other_personas: dict[str, "FishingPersona"]

    converse: FishingConverseComponent
    act: FishingActComponent

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        framework_model: ModelWandbWrapper,
        embedding_model: EmbeddingModel,
        base_path: str,
        memory_cls: type[AssociativeMemory] = AssociativeMemory,
        perceive_cls: type[PerceiveComponent] = PerceiveComponent,
        retrieve_cls: type[RetrieveComponent] = RetrieveComponent,
        store_cls: type[FishingStoreComponent] = FishingStoreComponent,
        reflect_cls: type[FishingReflectComponent] = FishingReflectComponent,
        plan_cls: type[FishingPlanComponent] = FishingPlanComponent,
        act_cls: type[FishingActComponent] = FishingActComponent,
        converse_cls: type[FishingConverseComponent] = FishingConverseComponent,
    ) -> None:
        super().__init__(
            cfg,
            model,
            framework_model,
            embedding_model,
            base_path,
            memory_cls,
            perceive_cls,
            retrieve_cls,
            store_cls,
            reflect_cls,
            plan_cls,
            act_cls,
            converse_cls,
        )

    def loop(self, obs: HarvestingObs) -> PersonaAction:
        res = []
        self.current_time = obs.current_time  # update current time

        self.perceive.perceive(obs)
        # phase based game

        if obs.current_location == "lake" and obs.phase == "lake":
            # Stage 1. Pond situation / Stage 2. Fishermen’s decisions
            retireved_memory = self.retrieve.retrieve([obs.current_location], 10)
            if obs.current_resource_num > 0:
                num_resource, html_interactions = self.act.choose_how_many_fish_to_chat(
                    retireved_memory,
                    obs.current_location,
                    obs.current_time,
                    obs.context,
                    range(0, obs.current_resource_num + 1),
                    obs.before_harvesting_sustainability_threshold,
                )
                action = PersonaActionHarvesting(
                    self.agent_id,
                    "lake",
                    num_resource,
                    stats={f"{self.agent_id}_collected_resource": num_resource},
                    html_interactions=html_interactions,
                )
            else:
                num_resource = 0
                action = PersonaActionHarvesting(
                    self.agent_id,
                    "lake",
                    num_resource,
                    stats={},
                    html_interactions="<strong>Framework<strong/>: no fish to catch",
                )
        elif obs.current_location == "lake" and obs.phase == "pool_after_harvesting":
            # dummy action to register observation
            action = PersonaAction(self.agent_id, "lake")
        elif obs.current_location == "restaurant":
            # Stage 3. Social Interaction a)
            # Need to first get the identities of the other personas that are in the restaurant
            other_personas_identities = []
            for agent_id, location in obs.current_location_agents.items():
                if location == "restaurant":
                    other_personas_identities.append(
                        self.other_personas_from_id[agent_id].identity
                    )

            (
                conversation,
                _,
                resource_limit,
                html_interactions,
            ) = self.converse.converse_group(
                other_personas_identities,
                obs.current_location,
                obs.current_time,
                obs.context,
                obs.agent_resource_num,
            )
            action = PersonaActionChat(
                self.agent_id,
                "restaurant",
                conversation,
                conversation_resource_limit=resource_limit,
                stats={"conversation_resource_limit": resource_limit},
                html_interactions=html_interactions,
            )
        elif obs.current_location == "home":
            # Stage 3. Social Interaction b)
            # TODO How what should we reflect, what is the initial focal points?
            self.reflect.run(["harvesting"])
            action = PersonaAction(self.agent_id, "home")

        self.memory.save()  # periodically save memory
        return action
