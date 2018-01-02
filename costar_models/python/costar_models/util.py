
from .multi_gan_model import RobotMultiGAN
from .multi_regression_model import RobotMultiFFRegression
from .multi_tcn_regression_model import RobotMultiTCNRegression
from .multi_lstm_regression import RobotMultiLSTMRegression
from .multi_conv_lstm_regression import RobotMultiConvLSTMRegression
from .multi_trajectory_sampler import RobotMultiTrajectorySampler
from .multi_autoencoder_model import RobotMultiAutoencoder
from .multi_hierarchical import RobotMultiHierarchical

# Model for sampling predictiosn
from .multi_sampler import RobotMultiPredictionSampler
from .goal_sampler import RobotMultiGoalSampler
from .multi_sequence import RobotMultiSequencePredictor
from .image_sampler import RobotMultiImageSampler
from .husky_sampler import HuskyRobotMultiPredictionSampler
from .pretrain_image import PretrainImageAutoencoder
from .pretrain_state import PretrainStateAutoencoder
from .pretrain_sampler import PretrainSampler
from .pretrain_minimal import PretrainMinimal
from .pretrain_image_gan import PretrainImageGan

from .sampler2 import PredictionSampler2
from .conditional_sampler2 import ConditionalSampler2

def MakeModel(features, model, taskdef, **kwargs):
    '''
    This function will create the appropriate neural net based on images and so
    on.

    Parameters:
    -----------
    features: string describing the set of features (inputs) we are using.
    model: string describing the particular method that should be applied.
    taskdef: a (simulation) task definition used to extract specific
             parameters.
    '''
    # set up some image parameters
    if features in ['rgb', 'multi']:
        nchannels = 3
    elif features in ['depth']:
        nchannels = 1

    model_instance = None
    model = model.lower()

    if features == 'multi':
        '''
        This set of features has three components that may be handled
        differently:
            - image input
            - current arm pose
            - current gripper state

        All of these models are expected to use the three fields:
            ["features", "arm", "gripper"]
        As a part of their state input.
        '''
        if model == 'gan':
            model_instance = RobotMultiGAN(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'predictor':
            model_instance = RobotMultiPredictionSampler(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'ff_regression':
            model_instance = RobotMultiFFRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'tcn_regression':
            model_instance = RobotMultiTCNRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'lstm_regression':
            model_instance = RobotMultiLSTMRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == 'conv_lstm_regression':
            model_instance = RobotMultiConvLSTMRegression(taskdef,
                    model=model,
                    **kwargs)
        elif model == "sample":
            model_instance = RobotMultiTrajectorySampler(taskdef,
                    model=model,
                    **kwargs)
        elif model == "autoencoder":
            model_instance = RobotMultiAutoencoder(taskdef,
                    model=model,
                    **kwargs)
        elif model == "hierarchical":
            model_instance = RobotMultiHierarchical(taskdef,
                    model=model,
                    **kwargs)
        elif model == "unsupervised":
            model_instance = RobotMultiUnsupervised(taskdef,
                    model=model,
                    **kwargs)
        elif model == "unsupervised1":
            model_instance = RobotMultiUnsupervised1(taskdef,
                    model=model,
                    **kwargs)
        elif model == "sequence":
            model_instance = RobotMultiSequencePredictor(taskdef,
                    model=model,
                    **kwargs)
        elif model == "husky_predictor":
            model_instance = HuskyRobotMultiPredictionSampler(taskdef,
                    model=model,
                    **kwargs)
        elif model == "goal_sampler":
            model_instance = RobotMultiGoalSampler(taskdef, model=model,
                    **kwargs)
        elif model == "image_sampler":
            model_instance = RobotMultiImageSampler(taskdef, model=model,
                    **kwargs)
        elif model == "pretrain_image_encoder":
            model_instance = PretrainImageAutoencoder(taskdef, model=model,
                    **kwargs)
        elif model == "pretrain_state_encoder":
            model_instance = PretrainStateAutoencoder(taskdef, model=model,
                    **kwargs)
        elif model == "pretrain_sampler":
            model_instance = PretrainSampler(taskdef, model=model, **kwargs)
        elif model == "predictor2" or model == "sampler2":
            model_instance = PredictionSampler2(taskdef, model=model, **kwargs)
        elif model == "conditional_sampler2":
            model_instance = ConditionalSampler2(taskdef, model=model, **kwargs)
        elif model == "pretrain_minimal":
            model_instance = PretrainMinimal(taskdef, model=model, **kwargs)
        elif model == "pretrain_image_gan":
            model_instance = PretrainImageGan(taskdef, model=model, **kwargs)
    
    # If we did not create a model then die.
    if model_instance is None:
        raise NotImplementedError("Combination of model %s and features %s" + \
                                  " is not currently supported by CTP.")

    return model_instance

def GetModels():
    return [None,
            "gan", # simple GAN model for generating images
            "ff_regression", # regression model; just a dense net
            "tcn_regression", # ff regression model with a TCN
            "lstm_regression", # lstm regression model
            "conv_lstm_regression", # lstm regression model
            "sample", # sampler NN to generate trajectories
            "predictor", # sampler NN to generate image goals
            "autoencoder", # autoencoder image test
            "hierarchical", # hierarchical policy for planning
            "unsupervised", # paper implementation for unsupervised method
            "unsupervised1", # alternative implementation
            "husky_predictor", # husky multi prediction sampler implementation
            "goal_sampler", # samples goals instead of everything else
            "image_sampler", #just learn to predict goal image
            "pretrain_image_encoder", # tool for pretraining images
            "pretrain_state_encoder", # tool for pretraining states
            "pretrain_sampler", # tool for pretraining the sampler
            "predictor2", # second version of the prediction-sampler code
            "sampler2", # -----------------------------   (same as above)
            "conditional_sampler2", # just give the condition
            "pretrain_minimal",
            "pretrain_image_gan",
            ]
