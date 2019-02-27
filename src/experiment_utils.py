import os
import logging
import torch
import numpy as np
from src import utils


def setup_model_logger(logger, model_name, log_root_path='logs/MIMIC3/'):
    
    
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    log_path = "{}/{}.log".format(log_root_path, model_name)
    
    if os.path.exists(log_path):
        os.remove(log_path)
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

def evaluate_validation_loss(model, loss_fn, test_inputs, test_outputs):
    model = model.eval()
    test_losses = []
    for sequence, target in zip(test_inputs, test_outputs):
        logits = model(sequence)
        loss = loss_fn( utils.to_tensor(logits[:-1]),
                utils.to_tensor(target[1:]))
        test_losses.append(loss.data)
    return test_losses

def train_model(model, loss_fn, optimizer, n_epoch, 
        train_inputs, train_outputs, logger=None): 
    model = model.train()
    logger.debug("training model")
    training_losses = []
    for epoch in range(n_epoch):
        curr_losses = []
        for sequence, target in zip(train_inputs,
                                    train_outputs):
            model.zero_grad()
            logits = model(sequence) 
            loss = loss_fn(utils.to_tensor(logits[1:]),
                    utils.to_tensor(target[1:]))
            curr_losses.append(loss.data)
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(curr_losses)
        training_losses.append(mean_loss)
        if not logger is None:
            logger.info("epoch: {}, loss: {}".format(epoch, mean_loss))
    return training_losses


def evaluate_validation_loss_template(model, 
        loss_fn, 
        inputs, 
        truth_outputs):
    model = model.eval()
    test_losses = []
    for x, target in zip(inputs, truth_outputs):
        output = model(utils.to_tensor(x))
        loss = loss_fn(output, utils.to_tensor(target))
        test_losses.append(loss.data)
    return test_losses

def train_model_template(model, loss_fn, optimizer, n_epoch,
        inputs, truth_outputs, logger = None):
    model = model.train()
    if not logger is None:
        logger.debug("Training model")
    training_losses = []
    for epoch in range(n_epoch):
        curr_losses = []
        for x, truth_output in zip(inputs, truth_outputs):
            model.zero_grad()
            x = utils.to_tensor(x)
            logits = model(x)
            loss = loss_fn(logits, utils.to_tensor(truth_output))
            curr_losses.append(loss.data)
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(curr_losses)
        training_losses.append(mean_loss)
        if not logger is None:
            logger.info("Epoch: {}, loss: {}".format(epoch, mean_loss))
    return training_losses
