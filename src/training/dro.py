import torch

def dro_train_epoch(model: torch.nn.Module, fold_loaders, 
                    optimizer: torch.optim.Optimizer, device):

    model.train()

    fold_iters = [iter(loader) for loader in fold_loaders]
    n_steps = min(len(loader) for loader in fold_loaders)

    total_loss = .0

    for _ in range(n_steps):

        optimizer.zero_grad()

        fold_losses = []

        for it in fold_iters:

            X, y = next(it)

            X: torch.Tensor = X.to(device)
            y: torch.Tensor = y.to(device)

            logits = model(X)
            loss = torch.nn.functional.cross_entropy(logits, y)

            fold_losses.append(loss)

        fold_losses = torch.stack(fold_losses)
        weights = torch.softmax(fold_losses.detach(), dim=0)
        dro_loss = torch.sum(weights * fold_losses)

        dro_loss.backward()
        optimizer.step()

        total_loss += dro_loss.item()

    return total_loss / n_steps