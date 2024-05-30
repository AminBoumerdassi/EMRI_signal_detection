import torch


'''Need to implement mixed precision in the training loop'''
def train_loop(dataloader, model, loss_fn, optimizer, batch_size, train_history, scaler, device, use_amp=True):
    train_loss=0.
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for X, y in dataloader:#batch, enumerate()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # Compute prediction and loss in float16 precision
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

        # Get scaled gradients from scaled losses
        scaler.scale(loss).backward()
        # Unscale the gradients
        scaler.step(optimizer)
        # Update the scale
        scaler.update()
        optimizer.zero_grad()

    
    print(f"Training loss: {train_loss:.6E}")#>7f

        # if batch % 100 == 0:
        #     train_loss, current = train_loss/size, batch * batch_size + len(X)
        #     print(f"Training loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")

    #Append loss to history
    train_history.append(train_loss)


def val_loop(dataloader, model, loss_fn, val_history, scaler, device, use_amp=True):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    #num_batches = len(dataloader)
    val_loss = 0.

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:#batch, enumerate()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                
    '''I assume that no loss scaling is needed here'''
    #Reduction method: dividing by ???
    val_loss /= size
    print(f"Validation loss: {val_loss:.6E}")#>8f

    #Append val. history
    val_history.append(val_loss)