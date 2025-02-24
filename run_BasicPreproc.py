# %% imports
from pathlib import Path
import mne
from almkanal import AlmKanal
from plus_slurm import Job, JobCluster, PermuteArgument
from cluster_jobs.c01_basic_preproc import BasicPreproc



#%% get subject IDs and Paths


data_path = '/home/sdiny/sd_therapy'
output_path = '/home/sdiny/sd_therapy_preproc'

if not Path(output_path).exists():
    Path(output_path).mkdir()
all_files = list(Path('/home/sdiny/sd_therapy').glob('*/*.fif'))
all_subjects = [str(file).split('/')[-1][:12] for file in all_files]
all_subjects = list(set(all_subjects))
all_subjects.sort()

#%% pilots
#all_subjects = all_subjects[1:2]

#%% delta from shorter resting (missing triggervalue for END)
all_subjects = ['19970203urmr', '19971127brms', '19980823dngu', '20000813cake', '20000926drsr', '20000928mrzm', '20001213isld', '20050228cash', '20050629hibe']


#%% create and submit jobs

job_cluster = JobCluster(
            required_ram= '16G',
            request_cpus= 4,
            request_time= 60*24,
            python_bin='/home/sdiny/masterthesis/sd_therapy/.pixi/envs/default/bin/python')


job_cluster.add_job(BasicPreproc,
                    subject_id=PermuteArgument(all_subjects),
                    data_path=data_path,
                    out_path = output_path,
                    ica_method = 'fastica',
                    subjects_dir = '',#TODO: get and add freesurfer subjects_dir
                    empty_room_path = '',#TODO: add empty_room_path
                    do_maxfilt=False,#TODO: fix maxfilt
                    lp=100,
                    hp=0.1,)

#%%
job_cluster.submit(do_submit=True)

# %%
