import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# --- Load filtered sequences ---
with open("balanced_sequences_take2.json", "r") as f:
    data = json.load(f)

# --- Parameters ---
SEQ_LEN = 10
num_features = 63
label_map = {"Idle": 0, "Aim": 1, "Fire": 2}
num_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Filter sequences to exact SEQ_LEN ---
filtered_data = []
removed_counts = Counter()
for seq in data:
    if len(seq["landmarks"]) == SEQ_LEN:
        filtered_data.append(seq)
    else:
        removed_counts[seq["label"]] += 1

print("Sequences removed due to length != 10:", removed_counts)
print("Sequences kept per class:", Counter(seq['label'] for seq in filtered_data))

# --- Stratified train/validation split ---
labels = [seq['label'] for seq in filtered_data]
train_idx, val_idx = train_test_split(
    list(range(len(filtered_data))), test_size=0.2, stratify=labels, random_state=42
)
train_dataset = [filtered_data[i] for i in train_idx]
val_dataset = [filtered_data[i] for i in val_idx]

print("Train counts:", Counter(seq['label'] for seq in train_dataset))
print("Val counts:", Counter(seq['label'] for seq in val_dataset))

# --- Dataset & DataLoader ---
class GestureDataset(Dataset):
    def __init__(self, sequences, label_map):
        self.sequences = sequences
        self.label_map = label_map

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_entry = self.sequences[idx]
        landmarks = np.array(seq_entry["landmarks"], dtype=np.float32)
        label = self.label_map[seq_entry["label"]]
        return torch.from_numpy(landmarks), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    X = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return X, y

train_loader = DataLoader(GestureDataset(train_dataset, label_map), batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(GestureDataset(val_dataset, label_map), batch_size=8, shuffle=False, collate_fn=collate_fn)

# --- TCN Model ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCNGestureClassifier(nn.Module):
    def __init__(self, num_features=63, num_classes=3, num_blocks=3, hidden_channels=64, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        dilations = [2**i for i in range(num_blocks)]
        in_ch = num_features
        for d in dilations:
            layers.append(ResidualBlock(in_ch, hidden_channels, kernel_size, dilation=d, dropout=dropout))
            in_ch = hidden_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = x.transpose(1,2)  # [batch, features, seq_len]
        x = self.network(x)
        x = x[:,:,-1]
        return self.fc(x)

model = TCNGestureClassifier(num_features=num_features, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Class-wise accuracy ---
def compute_class_accuracy(preds, labels, num_classes):
    correct = [0]*num_classes
    total = [0]*num_classes
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    return [c/t if t>0 else 0.0 for c,t in zip(correct,total)]

if __name__ == "main":
    # --- Training loop with best model saving ---
    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())

        class_acc = compute_class_accuracy(all_preds, all_labels, num_classes)
        overall_acc = sum([c * t for c, t in zip(class_acc, [all_labels.count(i) for i in range(num_classes)])]) / len(
            all_labels)
        print(f"Epoch {epoch + 1}/{num_epochs} | Overall Acc: {overall_acc:.4f} | Class-wise: {class_acc}")

        # --- Save best model ---
        if overall_acc > best_val_acc:
            best_val_acc = overall_acc
            torch.save(model.state_dict(), "tcn_gesture_model_best.pth")
            print(f"Best model saved at epoch {epoch + 1} with val accuracy {best_val_acc:.4f}")
