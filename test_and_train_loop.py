import torch

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, train_history, scaler, device, use_amp=True):
    train_loss=0.#Sum of losses across all batches
    no_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    #Set up a normalising tensor of maximum absolutes for our inputs
    max_abs_tensor= torch.as_tensor([0.9098072, 0.5969127], device="cuda").reshape(2,1)
    std_tensor= torch.as_tensor([0.0089, 0.0087], device="cuda").reshape(2,1)

    for X, y in dataloader:#batch, enumerate()
        with torch.autocast(device_type=device, dtype=torch.float32, enabled=use_amp):#dtype=torch.float16
            #Normalise input data
            X=X/max_abs_tensor#/std_tensor
            y=y#/std_tensor

            # Compute prediction and loss in float16 precision
            pred = model(X)
            loss = loss_fn(pred, y)
            #Sum the current batch's loss with the sum of previous batch losses
            train_loss += loss.item()

        # Get scaled gradients from scaled losses
        scaler.scale(loss).backward()
        # Unscale the gradients
        scaler.step(optimizer)
        # Update the scale
        scaler.update()
        optimizer.zero_grad()

    #Report average per-batch loss by dividing by no. of batches
    final_loss= train_loss/no_batches
    print(f"Training loss: {final_loss:.6E}")#>7f

    #Append loss to history
    train_history.append(final_loss)


def val_loop(dataloader, model, loss_fn, val_history, scaler, device, use_amp=True):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    no_batches = len(dataloader)
    val_loss = 0.

    #Set up a normalising tensor of maximum absolutes for our inputs
    max_abs_tensor= torch.as_tensor([0.9098072, 0.5969127], device="cuda").reshape(2,1)
    std_tensor= torch.as_tensor([0.0089, 0.0087], device="cuda").reshape(2,1)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:#batch, enumerate()
            #Normalise input data
            X=X/max_abs_tensor#/std_tensor
            y=y#/std_tensor

            with torch.autocast(device_type=device, dtype=torch.float32, enabled=use_amp):#dtype=torch.float16
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
    
    #Report average per batch loss
    final_loss= val_loss/no_batches
    print(f"Validation loss: {final_loss:.6E}")#>8f

    #Append val. history
    val_history.append(final_loss)