import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import SentimentDataset
from src.model import SentimentClassifier
import os
import argparse

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='code/data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--------------------------------------------------")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached:    {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
    print(f"--------------------------------------------------")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    train_dataset = SentimentDataset(
        os.path.join(args.data_dir, 'train.csv'), 
        tokenizer
    )
    test_dataset = SentimentDataset(
        os.path.join(args.data_dir, 'test.csv'), 
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    model = SentimentClassifier(n_classes=3)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    
    # Simple scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_steps//2, gamma=0.1)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_accuracy = 0
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_dataset)
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
            model,
            test_loader,
            loss_fn,
            device,
            len(test_dataset)
        )
        
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
            
if __name__ == '__main__':
    main()
