# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Thomas Hartmann
#
# This file is part of the plus_slurm Project,
# see: https://gitlab.com/thht/plus-slurm
#
#    plus_slurm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    plus_slurm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with obob_subjectdb. If not, see <http://www.gnu.org/licenses/>.
import datetime
import os
import resource
import socket
import sys
from pathlib import Path

import psutil

os.chdir('{{ working_directory }}')

sys.path.append(str(Path.cwd()))

requested_ram = {{required_mem}} / 1024  # noqa

from plus_slurm.job import JobItem  # noqa

if __name__ == '__main__':
    job_info = {
        'ClusterId': 0,
        'ProcId': 0,
        'requested_ram': requested_ram,
    }

    slurm_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])  # type: ignore
    slurm_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))  # type: ignore

    additional_path = '{{ append_to_path }}'
    if additional_path:
        sys.path.append(additional_path)

    job_fname = Path('{{ jobs_dir }}', 'slurm', f'job{slurm_task_id:03d}.json.gzip')

    job_item = JobItem(job_fname)

    job_object = job_item.make_object()

    job_started = datetime.datetime.now()

    psutil_process = psutil.Process()
    psutil_info = psutil_process.as_dict()

    print(f'Running on: {socket.gethostname()}')
    print(f'Running on CPUs: {psutil_info["cpu_affinity"]}')
    print(f'Now running {job_item}')
    print(f'Parameters: {job_item.args}')
    print(f'Keyword Parameters: {job_item.kwargs}')
    print(f'Job ID: {slurm_job_id}, Task ID: {slurm_task_id}')

    print(f'Starting Job at {job_started}\n##########', flush=True)
    job_object.run_private()
    job_stopped = datetime.datetime.now()
    print(f'##########\nJob stopped at {job_stopped}')
    print(f'Execution took {job_stopped - job_started}')

    initial_cpu_time = sum(psutil_info['cpu_times'])
    psutil_info = psutil_process.as_dict()
    final_cpu_time = sum(psutil_info['cpu_times'])

    try:
        avg_amount_of_cpus_used = (final_cpu_time - initial_cpu_time) / (job_stopped - job_started).seconds  # noqa
    except ZeroDivisionError:
        avg_amount_of_cpus_used = 1

    mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    mem_toomuch = 100 * (requested_ram - mem_used) / mem_used

    print(f'Your job used an average of {avg_amount_of_cpus_used:.2f} CPUs')
    print(f'Your job asked for {requested_ram:.2f}GB of RAM')
    print(f'Your job used a maximum of {mem_used:.2f}GB of RAM')
    print(f'You overestimated you memory usage by {mem_toomuch:.2f}%.')
