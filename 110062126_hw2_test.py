import torch
from torch import nn
from torchvision import transforms as T

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np
import os, time
from collections import deque

class NoisyNetLayer(nn.Module):
    """
    NoisyNet layer, factorized version
    """
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()

        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.cached_bias = None
        self.cached_weight = None

        self.register_noise_buffers()
        self.parameter_initialization()
        self.sample_noise()

    def forward(self, x, sample_noise=True):
        """
        Forward pass the layer. If training, sample noise depends on sample_noise.
        Otherwise, use the default weight and bias.
        """
        return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

    def register_noise_buffers(self) -> None:
        """
        Registers factorised noise buffers, creating empty tensors for input and
        output noise components with dimensions matching input and output features.
        """
        self.register_buffer(name='epsilon_input', tensor=torch.empty(self.in_features))
        self.register_buffer(name='epsilon_output', tensor=torch.empty(self.out_features))

    def _calculate_bound(self):
        """
        Determines the initialization bound for the FactorisedNoisyLayer based on the inverse
        square root of the number of input features. This approach to determining the bound
        takes advantage of the factorised noise model's efficiency and aims to balance the
        variance of the outputs relative to the variance of the inputs. Ensuring that the
        initialization of weights does not saturate the neurons and allows for stable
        gradients during the initial phases of training.

        Returns:
            float: The calculated bound for initializing the layer's parameters, optimized
            for the factorised noise model to encourage stability and efficiency in
            parameter updates during the onset of learning.
        """
        return self.in_features**(-0.5)

    @property
    def weight(self):
        """
        Computes and returns the noisy weights by applying factorised noise to
        the mean weight values through an outer product of input and output noises.

        Returns:
            torch.Tensor: The noisy weights computed using factorised noise.
        """
        if self.cached_weight is None:
            self.cached_weight = self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight
        return self.cached_weight

    @property
    def bias(self):
        """
        Computes and returns the noisy biases by applying output-side factorised noise
        to the mean bias values.

        Returns:
            torch.Tensor: The noisy biases computed using output-side factorised noise.
        """
        if self.cached_bias is None:
            self.cached_bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return self.cached_bias

    def sample_noise(self):
        """
        Samples factorised noise for both inputs and outputs using a standard normal
        distribution and applies a transformation to achieve the desired noise distribution.
        Resets cached weights and biases to ensure fresh computation with new noise.
        """
        with torch.no_grad():
            epsilon_input = torch.randn(self.in_features, device=self.epsilon_input.device)
            epsilon_output = torch.randn(self.out_features, device=self.epsilon_output.device)
            self.epsilon_input = (epsilon_input.sign() * torch.sqrt(torch.abs(epsilon_input))).clone()
            self.epsilon_output = (epsilon_output.sign() * torch.sqrt(torch.abs(epsilon_output))).clone()
        self.cached_weight = None
        self.cached_bias = None

    def parameter_initialization(self) -> None:
        """
        Initializes the parameters of the layer by setting the standard deviation of the
        noise and uniformly initializing the mean values within a bound derived from the
        inverse square root of the input features.
        """
        bound = self._calculate_bound()
        self.sigma_bias.data.fill_(value=self.sigma * bound)
        self.sigma_weight.data.fill_(value=self.sigma * bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.mu_weight.data.uniform_(-bound, bound)

class DDQN(nn.Module):
    """
    Dueling DQN implementation.

    input_dim : state_dim (c, h, w)
    output_dim : output_dim for CNN, should be flattened
    """

    def __init__(self, input_dim, output_dim=256, n_actions=12, lr=0.0001, name='DDQN.ckpt'):
        super(DDQN, self).__init__()
        self.device = "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ckpt_name = name

        self.cnn = self.build_cnn(input_dim[0]) # Just give #of channels
        # self.value = nn.Linear(output_dim, 1)
        # self.advantage = nn.Linear(output_dim, n_actions)
        self.value = NoisyNetLayer(output_dim, 1)
        self.advantage = NoisyNetLayer(output_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def forward(self, input):
        flat_val = self.cnn(input)
        value = self.value(flat_val)
        advantage = self.advantage(flat_val)
        return value, advantage

    def build_cnn(self, c):
        cnn = torch.nn.Sequential()
        # Convolution 1
        conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=4)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("conv_1", conv1)
        cnn.add_module("relu_1", nn.ReLU())

        # Convolution 2
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("conv_2", conv2)
        cnn.add_module("relu_2", nn.ReLU())

        # Convolution 3
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        nn.init.kaiming_normal_(conv3.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("conv_3", conv3)
        cnn.add_module("maxpool", nn.MaxPool2d(kernel_size=2))

        # Reshape CNN output
        class ConvReshape(nn.Module): forward = lambda self, x: x.view(x.size()[0], -1)
        cnn.add_module("reshape", ConvReshape())

        # Calculate input size
        state = torch.zeros(1, *(self.input_dim))
        dims = cnn(state)
        line_input_size = int(np.prod(dims.size()))

        # Linear 1
        line1 = nn.Linear(line_input_size, 512)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("line_1", line1)
        cnn.add_module("relu_4", nn.ReLU())

        # Linear 2
        line1 = nn.Linear(512, self.output_dim)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("line_2", line1)
        cnn.add_module("relu_5", nn.ReLU())

        return cnn
    
    def load_ckpt(self, ckpt_dir):
        self.load_state_dict(torch.load(ckpt_dir))

class DDDQN(nn.Module):
    """
    Double DDQN implementation
    Target Network will be update with soft approach

    input_dim : state_dim (c, h, w)
    output_dim : output_dim for CNN, should be flattened
    """
    def __init__(self, input_dim, output_dim=256, n_actions=12, ckpt_dir='./ckeckpoints/', name='DDDQN.ckpt'):
        super(DDDQN, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_filepath = os.path.join(self.ckpt_dir, name)
        self.device = "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.online_net = DDQN(input_dim=input_dim, output_dim=output_dim, n_actions=n_actions)
        self.target_net = DDQN(input_dim=input_dim, output_dim=output_dim, n_actions=n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.to(self.device)
    
    def forward(self, input):
        return self.target_net(input)
    
    def load_ckpt(self, save_dir):
        # save_dir = Path(self.ckpt_dir) / save_dir
        self.target_net.load_ckpt(save_dir)
        # self.online_net.load_ckpt(save_dir)

class Agent():
    def __init__(self):
        self.device = "cpu"
        state_dim = (4, 84, 84)
        self.net = DDDQN(input_dim=state_dim, ckpt_dir='./110062126_hw2_data.py')
        self.net.load_ckpt('./110062126_hw2_data.py')
        self.last_action = 0
        self.deque = deque(maxlen=4)
        self.action_space = [i for i in range(12)]
        self.counter = 1

    # [NOTE] old name: select_action
    def act(self, observation):
        """
        Choose action according to Advantage

        Inputs:
            observation (LazyFrame) : An observation of the current state
        Outputs:
            action_idx (int) : An integer representing the selected action
        """
        # preprocess
        ## Skip Frame
        if self.counter < 4:
            self.counter += 1
            return self.last_action
        self.counter = 0
        ## Gray Scale
        def permute_orientation(obs):
            # Since torchvision use [C, H, W] rather than [H, W, C], we should transform it first
            return T.ToTensor()(obs.astype('int64').copy())
        observation = permute_orientation(observation)
        observation = T.Grayscale()(observation)
        ## Downsample
        transforms = T.Compose(
            [T.Resize((84, 84) + (), antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation.float()).squeeze(0)

        ## Frame Stack
        self.deque.append(observation)
        if len(self.deque) != 4:
            return self.last_action
        observation = gym.wrappers.frame_stack.LazyFrames(list(self.deque))
        observation = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
        observation = torch.tensor(observation, device=self.device).unsqueeze(0)
        
        _, advantage = self.net.target_net.forward(observation)
        prob = nn.Softmax(dim=-1)(advantage/0.2)
        prob = prob.cpu().detach().numpy()[0]
        # action = np.argmax(prob)
        action = np.random.choice(self.action_space, p=prob)
        self.last_action = action
        return action

if __name__ == '__main__':
    # Create Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    agent = Agent()
    tot_reward = 0

    for i in range(50):
        r = 0
        done = False
        state = env.reset()
        start_time = time.time()

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            if time.time() - start_time > 120:
                break

            tot_reward += reward
            r += reward
            state = next_state
            # env.render()
        print(f'Game #{i}: {r}')
    env.close()
    print(f'mean_reward: {tot_reward/50}')