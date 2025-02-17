import os
import shutil
import uuid

import hydra
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

import wandb
from pathfinder import get_model
from simulation.utils import ModelWandbWrapper, WandbLogger

from .persona import EmbeddingModel
from .scenarios.fishing.run import run as run_scenario_fishing
from .scenarios.pollution.run import run as run_scenario_pollution
from .scenarios.sheep.run import run as run_scenario_sheep


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    This is the main method of the program. This is what you run to start the simulation.
    NOTE: the arguments passed to this method in terminal can be found in simulation/conf/config.yaml
    """
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Setting up the results logging in Wandb and locally
    # Locally stored results can be found in results/{experiment_name}/{run_name}, e.g. simulation/results/fishing_v7.0/delicious-biscuit-99)
    logger = WandbLogger(cfg.experiment.name, OmegaConf.to_object(cfg), debug=cfg.debug)
    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/{cfg.experiment.name}/{logger.run_name}",
    )

    # This is the case if we are using a single model type, where all agents use the same model (i.e. mix_llm not set or mix_llm=0)
    if len(cfg.mix_llm) == 0:
        model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)

        wrapper = ModelWandbWrapper(
            model,
            render=cfg.llm.render,
            wanbd_logger=logger,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            seed=cfg.seed,
            is_api=cfg.llm.is_api,
        )
        wrappers = [wrapper] * cfg.experiment.personas.num
        wrapper_framework = wrapper

    # This is the case where agents can have different backends for their model. You MUST set mix_llm to be the same as the number of agents.
    else:
        if len(cfg.mix_llm) != cfg.experiment.personas.num:
            raise ValueError(
                f"Length of mix_llm should be equal to personas.num: {cfg.experiment.personas.num}"
            )
        unique_configs = {}
        wrappers = []

        for idx, llm_config in enumerate(cfg.mix_llm):
            llm_config = llm_config.llm
            config_key = (
                llm_config.path,
                llm_config.is_api,
                llm_config.backend,
                llm_config.temperature,
                llm_config.top_p,
                llm_config.gpu_list,
            )
            if config_key not in unique_configs:
                # Initialize the model only if its config is not already in the unique set
                model = get_model(
                    llm_config.path,
                    llm_config.is_api,
                    cfg.seed,
                    llm_config.backend,
                    llm_config.gpu_list,
                )
                wrapper = ModelWandbWrapper(
                    model,
                    render=llm_config.render,
                    wanbd_logger=logger,
                    temperature=llm_config.temperature,
                    top_p=llm_config.top_p,
                    seed=cfg.seed,
                    is_api=llm_config.is_api,
                )
                unique_configs[config_key] = wrapper
            wrappers.append(unique_configs[config_key])

        # Initializes the Framework agent, which acts as a summarizer for human-readable purposes
        llm_framework_config = cfg.framework_model
        config_key = (
            llm_framework_config.path,
            llm_framework_config.is_api,
            llm_framework_config.backend,
            llm_framework_config.temperature,
            llm_framework_config.top_p,
            llm_framework_config.gpu_list,
        )
        if config_key not in unique_configs:
            model = get_model(
                llm_framework_config.path,
                llm_framework_config.is_api,
                cfg.seed,
                llm_framework_config.backend,
                llm_framework_config.gpu_list,
            )
            wrapper_framework = ModelWandbWrapper(
                model,
                render=llm_framework_config.render,
                wanbd_logger=logger,
                temperature=llm_framework_config.temperature,
                top_p=llm_framework_config.top_p,
                seed=cfg.seed,
                is_api=llm_framework_config.is_api,
            )
            unique_configs[config_key] = wrapper_framework
        else:
            wrapper_framework = unique_configs[config_key]

    # This is the class which deals with converting text into machine-understandable embeddings
    embedding_model = EmbeddingModel(device="cpu")

    # This is where you can choose the scenario(s) to run. PLEASE choose fishing
    if cfg.experiment.scenario == "fishing":
        run_scenario_fishing(
            cfg.experiment,
            logger,
            wrappers,
            wrapper_framework,
            embedding_model,
            experiment_storage,
        )
    elif cfg.experiment.scenario == "pollution":
        run_scenario_pollution(
            cfg.experiment,
            logger,
            wrappers,
            wrapper_framework,
            embedding_model,
            experiment_storage,
        )
    else:
        raise ValueError(f"Unknown experiment.scenario: {cfg.experiment.scenario}")

    # The final stuff at the end is all Hydra configurations
    # I do not know how this works, so best not to touch it I guess
    hydra_log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    shutil.copytree(f"{hydra_log_path}/.hydra/", f"{experiment_storage}/.hydra/")
    shutil.copy(f"{hydra_log_path}/main.log", f"{experiment_storage}/main.log")
    # shutil.rmtree(hydra_log_path)

    artifact = wandb.Artifact("hydra", type="log")
    artifact.add_dir(f"{experiment_storage}/.hydra/")
    artifact.add_file(f"{experiment_storage}/.hydra/config.yaml")
    artifact.add_file(f"{experiment_storage}/.hydra/hydra.yaml")
    artifact.add_file(f"{experiment_storage}/.hydra/overrides.yaml")
    wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
