import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from models.CustomDataset import CustomDataset
import utils.helpers as helpers
import time


def train(model, data_dir, train_cities, output_dir, device, batch_size, num_epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    dataset = CustomDataset(data_dir, train_cities)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for image, mask in data_loader:
            image = image.to(device)
            mask = mask.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(image)

            # Calculate loss
            loss = criterion(pred, mask.float())
            epoch_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(data_loader)

        now = time.time()
        print(f""
              f"Epoch: {epoch + 1}/{num_epochs} | "
              f"Loss: {epoch_loss:.5f} | "
              f"Time elapsed: {helpers.format_elapsed_time(now - start)}"
              )

    # Elapsed time
    end = time.time()
    print(f"Total time elapsed: {helpers.format_elapsed_time(end - start)}")

    # Save model file
    current_time = datetime.now()
    formatted_time = current_time.strftime('%d-%m-%Y_%H-%M-%S')
    trained_model_file = output_dir + formatted_time + '_' + str(batch_size) + '-' + str(num_epochs) + '.pth'
    torch.save(model.state_dict(), trained_model_file)
    print(f"Model saved to {trained_model_file}")

    return trained_model_file


def segment(model, model_file, device, image_path):
    model.load_state_dict(torch.load(model_file))
    model.eval()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = torch.from_numpy(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)

    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    output = helpers.labels2mask(pred)

    return output
