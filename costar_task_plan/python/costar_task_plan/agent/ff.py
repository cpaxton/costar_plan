from abstract import AbstractAgent

from costar_models import MakeModel

class FeedForwardAgent(AbstractAgent):
    '''
    Simple feed forward agent. Loads everything based on model definition and
    executes in the environment.

    This does not perform any checks for what kind of model you create -- the
    only thing is that it will create the model and use model.predict() to get
    two outputs.
    
    One output is expected to be the arm position command, the other the
    gripper position command.
    '''

    name = "random"

    def __init__(self, env, *args, **kwargs):
        super(FeedForwardAgent, self).__init__(env=env, *args, **kwargs)
        self.model = MakeModel(taskdef=self.env.task, **kwargs)
        self.model.load(self.env.world)

    def fit(self, num_iter):

        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            features = self.env.reset()

            # Make sure the model's state is clear before moving on to the next
            # step.
            self.model.reset()

            while not self._break:
                arm_cmd, gripper_cmd = self.model.predict(self.env.world)
                control = self.env.step(arm_cmd=arm_cmd[0],
                        gripper_cmd=gripper_cmd[0])
                features, reward, done, info = self.env.step(control)
                self._addToDataset(self.env.world,
                        control,
                        features,
                        reward,
                        done,
                        i,
                        self.model.name)
                if done:
                    break

            if self._break:
                return
        
