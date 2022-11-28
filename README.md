# Analysis of image pre-processing for playing Flappy Bird with DQN [CSCE-689].

<img src="demo/teaser.gif" width=110 align="right"><br/>

We analyze the effect of different types of image pre processing techniques on training an RL agent. The task is to learn to Flappy Bird using Deep Q-Learning. You can find more details about the experiments in our report [link to report].


## Environment Setup
```bash
conda create -n birdRL pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
conda activate birdRL
conda install tensorboardX
pip install pygame==1.9.4 opencv-python==3.4.4.19
```

## Training
The folders `Binary`, `Flow`, etc. are the experiments with different image pre-processing. `cd` to any folder and run the following command.
```bash
cd Binary
python train.py
```
You can check the tensorboard logs using:
```bash
tensorboard --logdir .
```

# Testing
Similar to training:
```bash
cd Binary
python test.py
```
