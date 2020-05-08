"""
Generate shell scripts for training the models on the SNLI dataset using the Metacentrum using PBS
"""
from network.utils.constants import ALL_MODELS_CONFIG, LOSSES
import os

template = """#!/bin/bash
#PBS -N {}
#PBS -l select=1:ncpus=3:ngpus=1:gpu_cap=cuda75:mem=24gb:scratch_local=30gb
#PBS -l walltime=16:00:00
#PBS -q gpu
#PBS -m abe

MODEL_NAME={}
LOSS={}

DATADIR=/storage/brno3-cerit/home/pasekj/SiameseSearch_BP

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-2.0.0-gpu-python3

export PYTHONUSERBASE=/storage/brno3-cerit/home/pasekj/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=$SCRATCHDIR/SiameseSearch_BP:$PYTHONPATH

test -n "$SCRATCHDIR" || {{ echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }}

cp -r $DATADIR $SCRATCHDIR || {{ echo >&2 "Error while copying input file(s)!"; exit 2; }}

cd $SCRATCHDIR/SiameseSearch_BP

python3 network/snli_baseline.py --model $MODEL_NAME --loss $LOSS > $SCRATCHDIR/stdout.txt || {{ echo &2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }}

mkdir $DATADIR/out/logs/snli
mkdir $DATADIR/out/checkpoints/snli
mkdir $DATADIR/out/console/snli
mkdir $DATADIR/out/logs/snli/$MODEL_NAME"_$LOSS"
mkdir $DATADIR/out/checkpoints/snli/$MODEL_NAME"_$LOSS"
mkdir $DATADIR/out/console/snli/$MODEL_NAME"_$LOSS"

cp -r $SCRATCHDIR/SiameseSearch_BP/network/logs/snli $DATADIR/out/logs/fit/snli/$MODEL_NAME"_$LOSS"/ && export CLEAN_SCRATCH=true || CLEAN_SCRATCH=false
cp -r $SCRATCHDIR/SiameseSearch_BP/network/checkpoints/snli $DATADIR/out/checkpoints/snli/$MODEL_NAME"_$LOSS"/ && export CLEAN_SCRATCH=true || CLEAN_SCRATCH=false
cp -r $SCRATCHDIR/stdout.txt $DATADIR/out/console/snli/$MODEL_NAME"_$LOSS"/ && export CLEAN_SCRATCH=true || CLEAN_SCRATCH=false
clean_scratch
"""


if __name__ == '__main__':
    i = 0
    os.mkdir("generated")
    for loss in LOSSES.keys():
        for model, config in ALL_MODELS_CONFIG.items():
            if config[1] == 3 and not config[2]:
                tmp = template.format(f"SNLI_{model}_{loss}", model, loss)
                with open(f"generated/{i}.sh", "w") as f:
                    f.write(tmp)
                i += 1
