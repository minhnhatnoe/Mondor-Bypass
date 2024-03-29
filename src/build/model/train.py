import torch
from torch import nn
from tqdm import tqdm
from . import models, loader
from sklearn.model_selection import KFold
kf = KFold(n_splits=32, shuffle=True, random_state=42)

model = models.BasicCNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model = model.to(models.device)
criterion = criterion.to(models.device)

def train_epoch(train_idx) -> tuple[int, int]:
    total_loss = 0
    total_acc = 0
    for x, y in tqdm(loader.get_dataloader(train_idx)):
        x = x.to(models.device, dtype=torch.float)
        y = y.type(torch.LongTensor)
        y = y.to(models.device)

        output = model(x)
        batch_loss = criterion(output, y)
        total_loss += batch_loss
 
        acc = (output.argmax(dim=1) == y).sum()
        total_acc += acc

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return total_loss, total_acc

def test_epoch(test_idx) -> tuple[int, int]:
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for x, y in tqdm(loader.get_dataloader(test_idx)):
            x = x.to(models.device, dtype=torch.float)
            y = y.type(torch.LongTensor)
            y = y.to(models.device)

            output = model(x)
            batch_loss = criterion(output, y)
            total_loss += batch_loss

            acc = (output.argmax(dim=1) == y).sum()
            total_acc += acc
            
    return total_loss, total_acc

def train() -> None:
    min_loss = 10**15
    for fold, (train_idx, test_idx) in enumerate(kf.split(loader.datay)):
        total_loss_train, total_acc_train = train_epoch(train_idx)
        total_loss_tests, total_acc_tests = test_epoch(test_idx)
        print(
            f'''Epochs: {fold+1}
            | Train Loss: {total_loss_train / len(train_idx):.3f}
            | Train Accuracy: {total_acc_train / len(train_idx):.3f}
            | Test Loss: {total_loss_tests / len(test_idx):.3f}
            | Test Accuracy: {total_acc_tests / len(test_idx):.3f}'''
        )
        if min_loss > total_loss_tests / 64:
            min_loss = total_loss_tests / 64
            torch.save(model.state_dict(), "blob/CNN.pt")
            print(f"Saved! Improved to {min_loss}")
