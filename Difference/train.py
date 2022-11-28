"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import cv2
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=100000)
    parser.add_argument("--replay_memory_size", type=int, default=10000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models_proj")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()

    opt.log_path = os.path.join(opt.saved_path, opt.log_path)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    os.makedirs(opt.saved_path, exist_ok = True)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0

    avg_loss = 0
    avg_qval = 0
    avg_rwrd = 0

    max_score = 0

    a = time.time()

    p2d1 = (0, 0, 0, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    p2d2 = (0, 0, 0, 4) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    p2d3 = (0, 0, 0, 6) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    p2d = [p2d3, p2d2, p2d1]
    b   = [6, 4, 2]

    while iter < opt.num_iters:

        state_p = torch.pow(torch.abs(state[:, 3:] - 0.33 * state[:, 1:2])/255, 0.3)*255
        # if iter == 30:
        #     cv2.imwrite("{}.png".format(0), np.repeat(state[0, 1][:, :, None].cpu().numpy(), 3, axis=2))
        #     cv2.imwrite("{}.png".format(1), np.repeat(state[0, 3][:, :, None].cpu().numpy(), 3, axis=2))
        #     cv2.imwrite("{}.png".format(2), np.repeat(state_p[0, 0][:, :, None].cpu().numpy(), 3, axis=2))
        #     exit()

        prediction = model(state_p)[0]
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            # print("Perform a random action")
            action = randint(0, 1)
        else:

            action = torch.argmax(prediction).item()#[0]

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        
        out, out_next = [], []
        # print(next_state.shape)

        next_state_p = torch.pow(torch.abs(next_state[:, 3:] - 0.33 * next_state[:, 2:3])/255, 0.3)*255
        # print(state_p.shape, next_state_p.shape)
        # exit()

        replay_memory.append([state_p, action, reward, next_state_p, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1

        avg_loss += loss.item()
        avg_qval += torch.max(prediction)
        avg_rwrd += reward

        if max_score < game_state.score:
            max_score = game_state.score

        if (iter+1) % 100 == 0:
            
            print("Iteration: {}/{}, Loss: {:.4f}, Epsilon {:.4f}, Reward: {}, Q-value: {:.3f}, Max Score: {} time: {:.3f}".format(
                iter + 1,
                opt.num_iters,
                avg_loss/100,
                epsilon, avg_rwrd/100, avg_qval/100, max_score, (time.time()-a)))

            writer.add_scalar('Train/Loss', avg_loss/100, iter)
            writer.add_scalar('Train/Epsilon', epsilon, iter)
            writer.add_scalar('Train/Reward', avg_rwrd/100, iter)
            writer.add_scalar('Train/Q-value', avg_qval/100, iter)
            writer.add_scalar('Train/Max-Score', max_score, iter)

            avg_loss = 0
            avg_qval = 0
            avg_rwrd = 0

            a = time.time()

        if (iter+1) % (opt.num_iters//10) == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter+1))
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
