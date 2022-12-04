# NERSC Perlmutter

## How to connect through SSH on windows:

First time only:
- Download ``sshproxy.exe`` from ``/global/cfs/cdirs/mfa/NERSC-MFA``

Always (every 24 hours or until your session expires):
  - Run ``sshproxy.exe -u salvarc``  (change ``salvarc`` to your username). 
  This will ask you for your password concatenated with your 2FA/MFA code.

Now you can connect to Perlmutter through SSH, using ``nersckey.ppk`` as the private key file.
To connect with putty, do now:
  - ``pageant ~/.ssh/nersckey.ppk``
  - ``putty -pageant <username>@perlmutter-p1.nersc.gov``


## Useful commands

### Get a list of all the jobs
    
    sqs  # shows your current (scheduled) runs

Similar: 

    squeue --me  # or squeue -u <username>

### Job resource usage

    sstat -j 864934 -o JobID,MaxRSS  # shows the maximum memory usage (resident set size) of job 864934 

