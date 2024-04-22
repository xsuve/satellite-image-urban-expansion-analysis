import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from model.CustomDataset import CustomDataset
import utils.helpers as helpers
import time


def train(model, data_dir, train_cities, output_dir, device, batch_size, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    dataset = CustomDataset(data_dir, train_cities)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    start = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        now = time.time()
        print(f""
              f"Epoch: {epoch + 1}/{num_epochs} | "
              f"Loss: {epoch_loss / len(data_loader)} | "
              f"Time elapsed: {helpers.format_elapsed_time(now - start)}"
              )

    # Elapsed time
    end = time.time()
    print(f"Total time elapsed: {helpers.format_elapsed_time(end - start)}")

    # Save models file
    current_time = datetime.now()
    formatted_time = current_time.strftime('%d-%m-%Y_%H-%M-%S')
    trained_model_file = output_dir + formatted_time + '.pth'
    torch.save(model.state_dict(), trained_model_file)
    print(f"Model saved to {trained_model_file}")

    return trained_model_file


def segment(model, model_file, test_img):
    model.load_state_dict(torch.load(model_file))
    model.eval()

    test_img = helpers.process_img(test_img, ismask=False)

    with torch.no_grad():
        output = model(test_img.unsqueeze(0))

    output = torch.softmax(output, dim=2)
    _, predicted_class = torch.max(output, dim=1)
    predicted_class = predicted_class.squeeze().cpu().numpy()

    colored_output = np.zeros((predicted_class.shape[0], predicted_class.shape[1], 3), dtype=np.uint8)
    for class_label, color in helpers.LABEL_COLORS.items():
        colored_output[predicted_class == class_label] = color

    return colored_output
