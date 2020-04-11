MODEL="$1"
NUM_WORKERS="$2"
NUM_LAYERS="$3"
MAX_GENERATIONS="$4"

#source ~/.bash_profile
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pbt

if [ ! -d "./$NUM_LAYERS-$MODEL-population" ]; then
  mkdir ./$NUM_LAYERS-$MODEL-population
fi

python master.py --model $MODEL --num-workers $NUM_WORKERS --num-layers $NUM_LAYERS --max-generations $MAX_GENERATIONS > ./$NUM_LAYERS-$MODEL-population/logs.txt 2> ./$NUM_LAYERS-$MODEL-population/err.txt

conda deactivate
