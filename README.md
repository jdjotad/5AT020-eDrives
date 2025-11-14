# 5AT020-eDrives
CBL Assignment eDrives

Files:
- 'controllers.py': PI and MPC implementation
- 'job.slurm': Example of bash file to run in Snellius using SLURM scheduler for jobs (it works with python files, not notebooks). Run using:
```console
user@hostname:~/$ sbatch job.slurm
```
- 'environment.py': Implementation of the PMSM model using a Gymnasium template (Farama Foundation)
- 'pmsm_control.ipynb': Incomplete example to run PI, MPC, and DDPG
- 'requirements.txt': Required packages to run 'pmsm_control.ipynb'
- 'test_slurm.py': Example of python file to test 'job.slurm'
- 'utils.py': Complementary functions to run 'pmsm_control.ipynb'.
