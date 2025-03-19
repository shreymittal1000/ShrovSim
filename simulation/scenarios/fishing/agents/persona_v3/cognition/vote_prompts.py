from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    consider_identity_persona_prompt,
    conversation_to_string_with_dash,
    get_sytem_prompt,
    candidate_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)

def prompt_action_vote(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
):
    lm = model.start_chain(identity.name, "fishing_cognition_act", "choose_act_options")

    candidates = ["Pep", "Jose"]

    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\n"
        lm += f"Task: Given the previous rounds of fishing and discussions, who would you vote for as the person with the best fishing strategy?"
        lm += f"Vote 0 for {candidates[0]} and 1 for {candidates[1]}"
        lm += reasoning_steps_prompt()
        lm += f' Put the final answer after "Answer:", example "Answer: 1" if you wish to vote for {candidates[1]}. You MUST end your answer in this "Answer: 1" format.'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            stop_regex=f"tons",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)

    return option, lm.html()


def prompt_action_vote_candidate(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
):
    lm = model.start_chain(identity.name, "fishing_cognition_act", "choose_act_options")

    candidates = ["Pep", "Jose"]

    with user():
        lm += f"{candidate_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\n"
        lm += f"Task: Given the previous rounds of fishing and discussions, who would you vote for as the person with the best fishing strategy?"
        lm += f"Vote 0 for {candidates[0]} and 1 for {candidates[1]}"
        lm += reasoning_steps_prompt()
        lm += f' Put the final answer after "Answer:", example "Answer: 1" if you wish to vote for {candidates[1]}. You MUST end your answer in this "Answer: 1" format.'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            stop_regex=f"tons",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)

    return option, lm.html()