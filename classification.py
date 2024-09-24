import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

class SE3Transformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SE3Transformer, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# load the train and test data
train_df = pd.read_csv("dataset/champs-scalar-coupling/train.csv")
test_df = pd.read_csv("dataset/champs-scalar-coupling/test.csv")

# Display the first few rows of the train and test data
print(train_df.head())
print(test_df.head())

# Extract features and target from train data
X_train = train_df[["atom_index_0", "atom_index_1", "type"]]
y_train = train_df["scalar_coupling_constant"]

# Prepare test data
X_test = test_df[["atom_index_0", "atom_index_1", "type"]]

## Preprocessing pipeline
# use the OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['atom_index_0', 'atom_index_1']),
        ('cat', OneHotEncoder(), ['type'])
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

# Split the training data into training and validation sets
X_train_full, X_val, y_train_full, y_val = train_test_split(X_train_processed, y_train, test_size=0.3, random_state=42)

# Convert the split data to PyTorch tensors
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

# Create DataLoader for the full training set and validation set
train_full_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)
train_full_loader = DataLoader(train_full_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_dim = X_train_tensor.shape[1]
output_dim = 1
model = SE3Transformer(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_full_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_full_loader.dataset)
    train_losses.append(train_loss)
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Calculate additional metrics on the validation set
model.eval()
val_predictions = []
val_targets = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        val_predictions.extend(outputs.squeeze().numpy())
        val_targets.extend(y_batch.numpy())

# Calculate Mean Absolute Error and Root Mean Squared Error
mae = mean_absolute_error(val_targets, val_predictions)
rmse = mean_squared_error(val_targets, val_predictions, squared=False)

print(f'Validation MAE: {mae:.4f}')
print(f'Validation RMSE: {rmse:.4f}')

# Make predictions on the test data
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze().numpy()

# Plot the distribution of the test predictions
plt.figure(figsize=(10, 5))
plt.hist(test_predictions, bins=50, edgecolor='k')
plt.xlabel('Predicted Scalar Coupling Constant')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions')
plt.show()

# Convert predictions to DataFrame for submission
submission_df = pd.DataFrame({"id": test_df["id"], "scalar_coupling_constant": test_predictions})
submission_df.to_csv("predictions/submission.csv", index=False)
print(submission_df.head())