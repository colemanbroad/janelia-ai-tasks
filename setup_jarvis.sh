set -exo pipefail

cd /home/jl_fs/

## we can actually just copy-paste this into term after ssh connection

git clone https://github.com/colemanbroad/janelia-ai-tasks
git clone https://github.com/facebookresearch/dinov3
cd janelia-ai-tasks
pip install -r requirements.txt
mkdir dinoweights
sh dl_weights.sh 

add-apt-repository ppa:maveonair/helix-editor
apt update
apt install helix
