import sys
import os

# usage: xonsh eval_many.xsh <filename_substring_to_match> <grasp_model_string> <cuda_visible_devices>
# this xonsh shell script (https://github.com/xonsh/xonsh) i
# will run the evaluation step on a whole set of model files in a folder.
# all parameters are optional.
# NOTE: CUDA OPTION IS NOT YET WORKING 
# cuda_visible_devices can be "0" for gpu 0, "0,1" for gpus 0 and 1, "-1" for cpu only.

model_str = ''
cuda_dev = ''

if len(sys.argv) > 1:
    matchstr = sys.argv[1]
if len(sys.argv) > 2:
    model_str = sys.argv[2]
if len(sys.argv) > 3:
    cuda_dev = sys.argv[1]

if not model_str:
    model_str = 'grasp_model_levine_2016'
if not cuda_dev:
    cuda_dev = "\"0\""

listvar = $(ls).split('\n')
for weightsfile in reversed(listvar):
    # print(weightsfile)
    if '.h5' in weightsfile and matchstr in weightsfile and 'evaluation_dataset' not in weightsfile:
        print(weightsfile)
        # another way to print
        # $[echo @(weightsfile)]

        # debugging how to set cuda visible devices in xonsh
        #CUDA_VISIBLE_DEVICES="-1"
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        #$CUDA_VISIBILE_DEVICES = @(cuda_dev)
        #CUDA_VISIBLE_DEVICES = @(cuda_dev)
        CUDA_VISIBLE_DEVICES="1" && python grasp_train.py --epochs 1 --load_weights @(weightsfile) --pipeline_stage eval --grasp_model @(model_str)  --data_dir=~/datasets/grasping.ssd/
        print('evaluation complete for: ' + weightsfile)
