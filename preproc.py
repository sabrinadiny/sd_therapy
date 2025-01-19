# %% imports
from cluster_jobs.c01_basic_preproc import BasicPreproc
from pathlib import Path
from plus_slurm import JobCluster
from helpers.getSubjects import getSubjectsFrom


# %% get subject IDs and Paths

data_path = 'path/to/data'# path to fif files (plus_sync)
all_subjects = getSubjectsFrom(data_path, subfolder=True, extension='.fif')


#%% create and submit jobs

job_cluster = JobCluster(
            required_ram= '16G',
            request_cpus= 1,
            request_time= 120)

for subject in all_subjects:
    for block in range(1, 4): # assuming 4 blocks  
        job_cluster.add_job(BasicPreproc,
                            subject_id=subject,
							data_path=Path(data_path),
							block_nr=block,
							lp=100,
							hp=0.1,)
        job_cluster.submit()
