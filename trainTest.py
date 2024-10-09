import torch
import os
import yaml
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from datetime import datetime

from HsiDataloader import HyperspectralDataset, load_data

# def train_my_model(config, model):
def train_my_model(config, model):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    dataX, labelsY = load_data(config.SOLVER)
    trainset = HyperspectralDataset(dataX,labelsY,config)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=config.SOLVER['batch_size'], shuffle=True, num_workers=0
    )
    
    model.train()
    # put model on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # use Adam optimizer with learning rate 0.001
    optimizer = optim.Adam(model.parameters(), lr=config.SOLVER['lr'])

    # start training        
    print("Start Training")
    log_lines = []
    
    total_loss = 0
    for epoch in range(config.SOLVER['epochs']):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # input shape: (batch_size, C, H, W)
            outputs = model(inputs)
            
            if config.MODEL['removeZeroLabels']:
                labels -= 1
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 1 == 0:
            avg_loss = total_loss / (epoch + 1)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] [Epoch: {epoch + 1}]   [loss avg: {avg_loss:.5f}]   [current loss: {loss.item():.5f}]\n"
            print(log_line)
            log_lines.append(log_line)
    print("Finished Training")

    current_time = datetime.now().strftime("%y%m%d_%H-%M")
    log_file_path = os.path.join(config.OUTPUT_DIR, f'training_log_{current_time}.txt')
    config.SOLVER['log_path'] = log_file_path
    with open(log_file_path, 'w') as log_file:
        log_file.writelines(log_lines)
        log_file.write("Configuration:\n")
        yaml.dump(config.config, log_file)
        log_file.write("\n")

    
    model_save_path = os.path.join(config.OUTPUT_DIR, f'model_{current_time}.pth')
    torch.save(model.state_dict(), model_save_path)
    config.MODEL['weights'] = model_save_path
    config.SOLVER['mode'] = "test"
    print(f"Model saved to: {model_save_path}")
    test_my_model(config, model)


def test_my_model(config, model):
    dataX, labelsY = load_data(config.SOLVER)
    testset = HyperspectralDataset(dataX,labelsY,config)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=config.SOLVER['batch_size'], shuffle=True, num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_weights = config.MODEL['weights']
        
    model.load_state_dict(torch.load(model_weights, weights_only=True))
    model.to(device)
    model.eval()
    
    count = 0
    print("... ... Start Testing ... ...")
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    if config.MODEL['removeZeroLabels']:
        y_test = y_test - 1

    classification = classification_report(y_test, y_pred_test, digits=4)
    print(classification)
    with open(config.SOLVER['log_path'], 'a') as log_file:
        log_file.write(classification)

def class_predict(config, model):
    if config.MODEL['removeZeroLabels']:
        print("The model trained without zero labels is not supported to generate Image")
        return
    else:
        dataX, labelsY = load_data(config.SOLVER)
        trainset = HyperspectralDataset(dataX,labelsY,config)
        pridiction_loader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=1, shuffle=False, num_workers=0
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model_weights = config.MODEL['weights']
        model.load_state_dict(torch.load(model_weights, weights_only=True))
        model.eval()
        model.to(device)

        count = 0

        for inputs, _ in pridiction_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            if count == 0:
                prediction = outputs
                count = 1
            else:
                prediction = np.concatenate((prediction, outputs))

        plt.figure(figsize=(5, 5))
        plt.imshow(prediction.astype(int), cmap="jet")
        plt.colorbar()
        plt.title("Prediction")
        plt.show()
        print("... ... prediction done ... ...")