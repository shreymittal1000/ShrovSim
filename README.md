# ShrovSim - My Fork of GovSim
Note: all commands use time and slurm, as I log how much time each experiment took AND I run these on a slurm cluster

Example API commands we can use:
```
sbatch --gpus=1 --mem-per-cpu=8G  --wrap="time python -m simulation.main experiment=fish_baseline_concurrent llm.path=claude-3-5-sonnet-20240620 llm.is_api=true llm.backend=openrouter"
```
```
sbatch --gpus=1 --mem-per-cpu=8G  --wrap="time python -m simulation.main experiment=fish_baseline_concurrent llm.path=meta-llama/llama-3.1-8b-instruct llm.is_api=true llm.backend=openrouter"
```

Example Local Model commands we can use:
```
sbatch --gpus=1 --mem-per-cpu=8G  --wrap="time python -m simulation.main experiment=fish_baseline_concurrent"
```
```
sbatch --gpus=1 --mem-per-cpu=8G  --wrap="time python -m simulation.main experiment=fish_baseline_concurrent llm.path=llm.path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
```

# Slurmless API commands:
```
time python -m simulation.main experiment=fish_baseline_concurrent llm.path=claude-3-5-sonnet-20240620 llm.is_api=true llm.backend=openrouter
```
```
time python -m simulation.main experiment=fish_baseline_concurrent llm.path=meta-llama/llama-3.1-8b-instruct llm.is_api=true llm.backend=openrouter
```