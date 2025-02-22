#%%
from pathlib import Path
import joblib
import mne
from plus_slurm import Job
from almkanal import AlmKanal
import sys

data_path = '/home/aweigl/sd_therapy'
output_path = '/home/aweigl/sd_therapy_preproc'

if not Path(output_path).exists():
    Path(output_path).mkdir()
all_files = list(Path('/home/aweigl/sd_therapy').glob('*/*.fif'))
all_subjects = [str(file).split('/')[-1][:12] for file in all_files]
all_subjects = list(set(all_subjects))
all_subjects.sort()

subject_id='19970203urmr'

#%%

conditions = ['resting_eyes_closed','bodyscan', 'safeplace', 'podcast']
conditions = conditions[:1]
audio_lengths = {
    'bodyscan': 724,
    'safeplace': 645,
    'podcast': 578
}

event_id = {
'resting_eyes_closed': {'start': 10, 'end': 11},
'bodyscan': {'start': 21, 'end': 41},
'safeplace': {'start': 22, 'end': 42},
'podcast': {'start': 23, 'end': 43}
}

data = {}

#%%
for cond in conditions:
    full_path = list(Path(data_path).glob(f'*/*{subject_id}*{cond}*.fif'))[0]

    raw = mne.io.read_raw(full_path, preload=True)

    ak = AlmKanal(raw=raw)

    ak.raw.filter(l_freq=1, h_freq=100)

    try:
        events = mne.find_events(ak.raw)
    except ValueError:
        events = mne.find_events(ak.raw, shortest_event=1)



    event_id_dict = event_id[cond]
    if cond == 'resting_eyes_closed':
        if len(events) == 1:
            event_time =  (events[events[:, 2] == event_id_dict['start'], 0][0] - ak.raw.first_samp) / ak.raw.info['sfreq']
        else:
            event_times = [(events[events[:, 2] == event_id_dict['start'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq'], (events[events[:, 2] == event_id_dict['end'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq']]
    else:
        event_times = [(events[events[:, 2] == event_id_dict['start'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq'], (events[events[:, 2] == event_id_dict['start'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq']+audio_lengths[cond]]
        
    if len(events) == 1:
        ak.raw.crop(event_time)
    else:
        ak.raw.crop(tmin=event_times[0][0], tmax=event_times[1][0])


    if 'BIO003' in ak.raw.ch_names:
            ak.raw.set_channel_types({'BIO001': 'eog',
                                    'BIO002': 'eog',
                                    'BIO003': 'ecg',})

            mne.rename_channels(ak.raw.info, {'BIO001': 'EOG001',
                                            'BIO002': 'EOG002',
                                            'BIO003': 'ECG003',})

    ak.do_ica(n_components= None,
                method=ica_method,
                resample_freq= 200,  # downsample to 200hz per default
                eog =True,
                ecg = True,            
                train = True,
                train_freq = 16,
                )


    data.update({cond: 
        {'cropped_raw' : ak.raw}})

#joblib.dump(data, f'{out_path}/{subject_id}_preproc_data.dat')

# %%
