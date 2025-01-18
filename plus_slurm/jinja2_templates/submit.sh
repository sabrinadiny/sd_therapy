#!/bin/bash
{% set log_dir = jobs_dir / "log" %}
#SBATCH --cpus-per-task={{ request_cpus }}
#SBATCH --mem={{ required_mem }}
#SBATCH --array=1-{{ n_jobs }}
#SBATCH --time={{ request_time }}
#SBATCH --output={{ log_dir }}/out_%a.log
#SBATCH --export={{ export }}
{% if exclude_nodes %}
#SBATCH --exclude="{{ exclude_nodes }}"
{% endif %}
{% if qos %}
#SBATCH --qos={{ qos }}
{% endif %}
{% for slurm_arg in extra_slurm_args %}
#SBATCH {{ slurm_arg }}
{% endfor %}

echo "Executing Job $SLURM_JOB_ID on $SLURMD_NODENAME"
{{ executable }} {{ arguments }}