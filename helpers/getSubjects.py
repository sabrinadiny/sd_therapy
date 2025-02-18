import pathlib
import os
import glob

def getSubjectsFrom(directory, subfolder=False, extension='.fif'):
    """
    Returns a list of subject-ids from a given directory.
    The subject-ids are represented as the first 12 characters of the filename of each file in the given directory.
    :param directory: The directory to search for subject-ids.
    :param subfolder: Boolean. If True, the function will search for subject-ids in subfolders of the given directory.
    :param extension: The file extension to search for.
    """
    subject_ids = set()
    
    if subfolder:
        directory = os.path.join(directory + '/*/')
        
    for file in pathlib.Path(directory).glob('*' + extension):
        subject_id = file.name[:12]
        subject_ids.add(subject_id)
    return list(subject_ids)
