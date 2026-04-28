#!/bin/bash
# HOST="user@jarvis"
source .env
REMOTE_DIR="~/janelia-ai-tasks"
CONFIG="${1:-config.toml}"

rsync -av \
  --exclude='cache' \
  --exclude='dinoweights' \
  --exclude='.venv' \
  --exclude='.jj' \
  --exclude='.git' \
  --exclude='figures*' \
  ./ $HOST:$REMOTE_DIR/

ssh $HOST "cd $REMOTE_DIR && python main.py -c $CONFIG"

rsync -av $HOST:/home/janelia-ai-tasks/figures/*.png ./figures-remote/

# jl pause $JARVISMACHINE
