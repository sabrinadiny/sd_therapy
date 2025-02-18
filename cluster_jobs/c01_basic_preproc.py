# Just max- and bp- filters the data, does ICA and saves the preprocessed data
# for a given subject and block number.

from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import almkanal

class BasicPreproc(Job):
    job_data_folder = 'MEG_DATA'

    def run(
        self,
        subject_id: str,
        data_path: str,
        #subjects_dir: str,
        block_nr: int,
        #empty_room_path: str,
        lp: float = 100,
        hp: float = 0.1,
    ) -> None:
        full_path = Path(data_path) / subject_id + f'_block_{block_nr}.fif'
        raw = mne.io.read_raw(full_path, preload=True)
        
        # initialize almkanal object
        ak = almkanal(raw=raw)
        
        # raw preproc
        ak.do_maxwell()
        
        # apply bandpass filter
        ak.raw.filter(hp, lp)
        
        # do ICA
        ak.do_ica()
        
        # save the preprocessed data
        joblib.dump(ak, self.full_output_path)
            