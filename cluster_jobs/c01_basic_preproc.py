from pathlib import Path
import joblib
import mne
from plus_slurm import Job
from almkanal import AlmKanal
import sys

sys.path.append('/home/aweigl/msc/sd_therapy/cluster_jobs')
#from meta_job import Job

class BasicPreproc(Job):
    #job_data_folder = 'Basic_Preproc'

    def run(
        self,
        subject_id: str,
        data_path: str,
        out_path: str,
        subjects_dir: str,
        empty_room_path: str,
        do_maxfilt: bool = False,
        do_source: bool = False,
        ica_method: str='fastica',
        lp: float = 100,
        hp: float = 0.1,
    ) -> None:
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
            
            # apply maxfilt as soon as problems are fixed
            if do_maxfilt:
                ak.do_maxwell(mw_calibration_file='/home/aweigl/msc/data_sync/settings/sss_cal.dat',
                            mw_cross_talk_file='/home/aweigl/msc/data_sync/settings/ct_sparse.fif')
            
            ak.raw.filter(l_freq=hp, h_freq=lp)
            
            try:
                events = mne.find_events(ak.raw)
            except ValueError:
                events = mne.find_events(ak.raw, shortest_event=1)
            
            if cond == 'resting_eyes_closed':
                if len(events) == 1:
                    event_time =  (events[events[:, 2] == event_id[cond]['start'], 0][0] - ak.raw.first_samp) / ak.raw.info['sfreq']
                else:
                    event_times = [(events[events[:, 2] == event_id[cond]['start'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq'], (events[events[:, 2] == event_id[cond]['end'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq']]
            else:
                event_times = [(events[events[:, 2] == event_id[cond]['start'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq'], (events[events[:, 2] == event_id[cond]['start'], 0] - ak.raw.first_samp) / ak.raw.info['sfreq']+audio_lengths[cond]]
                
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
                        
            # ECG R-Peak Correction: Ensure correct indentation
            if 'ECG003' in ak.raw.ch_names:  # Check if the ECG channel exists
                ecg_epochs = mne.preprocessing.create_ecg_epochs(
                    ak.raw, 
                    ch_name='ECG003',  # Specify the ECG channel
                    tmin=-0.3, tmax=0.3,  # 300ms before and after each R-peak
                    baseline=(None, 0),  # Baseline correction from start of epoch
                    preload=True
                )

                # Compute the average ECG artifact
                ecg_evoked = ecg_epochs.average()

                # Subtract the ECG artifact from the raw MEG data
                ak.raw = ak.raw.copy().subtract_evoked(ecg_evoked)


            if do_source:
                ak.do_fwd_model(subject_id=subject_id,
                                    subjects_dir=subjects_dir, 
                                    redo_hdm=True)
                #%% go 2 source
                stc = ak.do_src(
                    subject_id=subject_id,
                    return_parc=True,
                    empty_room=empty_room_path,
                    get_nearest_empty_room=True,
                )
                
                data.update({cond: 
                    {'cropped_raw' : ak.raw,
                    'stc': stc}})
            else:
                data.update({cond: 
                    {'cropped_raw' : ak.raw}})

        joblib.dump(data, f'{out_path}/{subject_id}_preproc_data.dat')
            