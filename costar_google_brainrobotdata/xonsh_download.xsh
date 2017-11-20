# Download the grasp dataset with xonsh shell:
# cd /path/to/grasp_dataset.py
# xonsh xonsh_download.xsh
# source xonsh_download.xsh
# github.com/xonsh/xonsh

# to show trace when running with xonsh
# $XONSH_SHOW_TRACEBACK = True
import sys
sys.path.insert(0, '')
import grasp_dataset
g = grasp_dataset.GraspDataset(dataset='all')
g.download(dataset='all')
