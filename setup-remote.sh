#!/bin/bash
set -euo pipefail
source .env

REMOTE_DIR="~/janelia-ai-tasks"

rsync -av -e 'ssh -o StrictHostKeyChecking=no' \
  --exclude='cache' \
  --exclude='dinoweights' \
  --exclude='.venv' \
  --exclude='.jj' \
  --exclude='.git' \
  --exclude='figures*' \
  ./ $HOST:$REMOTE_DIR/

ssh -o StrictHostKeyChecking=no $HOST "cd $REMOTE_DIR && pip install -r requirements-headless.txt"
