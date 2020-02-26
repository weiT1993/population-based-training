MODEL="$1"
NUM_WORKERS="$2"
NUM_LAYERS="$3"
MAX_GENERATIONS="$3"

if [ ! -d "./$NUM_LAYERS-$MODEL-population" ]; then
  mkdir ./$NUM_LAYERS-$MODEL-population
fi

python master.py --model $MODEL --num-workers $NUM_WORKERS --num-layers $NUM_LAYERS --max-generations $MAX_GENERATIONS 2>&1 | tee ./$NUM_LAYERS-$MODEL-population/logs.txt
