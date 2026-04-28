#!/bin/bash
set -euo pipefail

source .env
CONFIG="${1:-config.toml}"

# rsync -av -e 'ssh -o StrictHostKeyChecking=no' \
#   --exclude='cache' \
#   --exclude='dinoweights' \
#   --exclude='.venv' \
#   --exclude='.jj' \
#   --exclude='.git' \
#   --exclude='figures*' \
#   ./ $HOST:$REMOTE_DIR/

# ssh -o StrictHostKeyChecking=no $HOST "cd $REMOTE_DIR && PYTHONUNBUFFERED=1 python main.py -c $CONFIG"
rsync -av -e 'ssh -o StrictHostKeyChecking=no' $HOST:/home/janelia-ai-tasks/figures/*.gif ./figures-remote/
