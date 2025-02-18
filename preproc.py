# %% imports
from pathlib import Path
import mne
from almkanal import AlmKanal



# %% get subject IDs and Paths

#data_path = '/home/aweigl/msc/data_sync/data_synced/sd_therapy'# path to fif files (plus_sync)
#all_subjects = getSubjectsFrom(data_path, subfolder=True, extension='.fif')

datadir = '/home/aweigl/msc/data_sync/data_synced/sd_therapy'
all_files = list(Path('/home/aweigl/msc/data_sync/data_synced/sd_therapy').glob('*/*.fif'))
all_subjects = [str(file).split('/')[-1][:12] for file in all_files]
all_subjects = list(set(all_subjects))


full_path = '/home/aweigl/msc/data_sync/data_synced/sd_therapy/250115/20050228cash_lssh28_condition1_bodyscan.fif'
raw = mne.io.read_raw(full_path, preload=True)

# initialize almkanal object
ak = AlmKanal(raw=raw)

#%% events
events = mne.find_events(ak.raw)

#%% raw preproc
ak.do_maxwell(mw_calibration_file='/home/aweigl/msc/data_sync/settings/sss_cal.dat',
              mw_cross_talk_file='/home/aweigl/msc/data_sync/settings/ct_sparse.fif')



#%%apply bandpass filter
ak.raw.filter(hp, lp)

# do ICA
ak.do_ica()

# Updated trigger mappings from Trigger_Values.txt
event_id = {
    'generalInstructions': 12,                  # General Instructions
    'finalInstructions': 99,                    # Final Instructions
    'restingStateStart': 10,                    # Resting State Start
    'restingStateEnd': 11,                      # Resting State End
    'preCondition1': 21 + 10,                   # Pre-Condition for Condition 1 (Mindfulness)
    'preCondition2': 22 + 10,                   # Pre-Condition for Condition 2 (Relaxation)
    'preCondition3': 23 + 10,                   # Pre-Condition for Condition 3 (Podcast)
    'condition1': 21,                           # Condition Audio Playback for Condition 1
    'condition2': 22,                           # Condition Audio Playback for Condition 2
    'condition3': 23,                           # Condition Audio Playback for Condition 3
    'endConditionIntro1': 21 + 20,              # End-Condition Intro for Condition 1
    'endConditionIntro2': 22 + 20,              # End-Condition Intro for Condition 2
    'endConditionIntro3': 23 + 20,              # End-Condition Intro for Condition 3
    'endConditionQuestions1': 21 + 1000,        # End-Condition Questions for Condition 1
    'endConditionQuestions2': 22 + 1000,        # End-Condition Questions for Condition 2
    'endConditionQuestions3': 23 + 1000,        # End-Condition Questions for Condition 3
    'betweenConditions1': 21 + 2000,            # Between Conditions for Condition 1
    'betweenConditions2': 22 + 2000,            # Between Conditions for Condition 2
    'betweenConditions3': 23 + 2000,            # Between Conditions for Condition 3
    'question1_1': 21 * 10 + 1,                 # Question 1 for Condition 1
    'question1_2': 21 * 10 + 2,                 # Question 2 for Condition 1
    'question1_3': 21 * 10 + 3,                 # Question 3 for Condition 1
    'question1_4': 21 * 10 + 4,                 # Question 4 for Condition 1
    'question1_5': 21 * 10 + 5,                 # Question 5 for Condition 1
    'question2_1': 22 * 10 + 1,                 # Question 1 for Condition 2
    'question2_2': 22 * 10 + 2,                 # Question 2 for Condition 2
    'question2_3': 22 * 10 + 3,                 # Question 3 for Condition 2
    'question2_4': 22 * 10 + 4,                 # Question 4 for Condition 2
    'question2_5': 22 * 10 + 5,                 # Question 5 for Condition 2
    'question3_1': 23 * 10 + 1,                 # Question 1 for Condition 3
    'question3_2': 23 * 10 + 2,                 # Question 2 for Condition 3
    'question3_3': 23 * 10 + 3,                 # Question 3 for Condition 3
    'question3_4': 23 * 10 + 4,                 # Question 4 for Condition 3
    'question3_5': 23 * 10 + 5,                 # Question 5 for Condition 3
    'nextQuestion1_1': 21 + 200 + 1,            # After Question 1 for Condition 1
    'nextQuestion1_2': 21 + 200 + 2,            # After Question 2 for Condition 1
    'nextQuestion1_3': 21 + 200 + 3,            # After Question 3 for Condition 1
    'nextQuestion1_4': 21 + 200 + 4,            # After Question 4 for Condition 1
    'nextQuestion2_1': 22 + 200 + 1,            # After Question 1 for Condition 2
    'nextQuestion2_2': 22 + 200 + 2,            # After Question 2 for Condition 2
    'nextQuestion2_3': 22 + 200 + 3,            # After Question 3 for Condition 2
    'nextQuestion2_4': 22 + 200 + 4,            # After Question 4 for Condition 2
    'nextQuestion3_1': 23 + 200 + 1,            # After Question 1 for Condition 3
    'nextQuestion3_2': 23 + 200 + 2,            # After Question 2 for Condition 3
    'nextQuestion3_3': 23 + 200 + 3,            # After Question 3 for Condition 3
    'nextQuestion3_4': 23 + 200 + 4,            # After Question 4 for Condition 3
}

# Extract events from the raw data (ensure that the event channel is properly marked in your data)
events = mne.find_events(ak.raw)

# Create epochs with a window from -0.2 to 0.5 sec (adjust as required)
epochs = mne.Epochs(ak.raw, events=events, event_id=event_id,
                    tmin=-0.2, tmax=0.5, baseline=(None, 0))

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
