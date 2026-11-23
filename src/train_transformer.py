import copy
import time
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from protgen.data.make_data import IGNORE_INDEX, VOCABULARY, VOCABULARY_SIZE, ProteinDataset
from protgen.models.mlp import CONTEXT_SIZE, MultiLayerPerceptron
from torch.utils.data import DataLoader

from protgen.models.transformer import Transformer

# for plotting loss curves 
results = {
    "Epoch_number": [],
    "validation_loss": [],
    "training_loss": []
}

LEARNING_RATE = 0.0003

def main():
    model = Transformer().to("cuda")

    train_data = ProteinDataset(path_to_data="data/train.csv")
    val_data = ProteinDataset(path_to_data="data/val.csv")
    test_data = ProteinDataset(path_to_data="data/test.csv")

    BATCH_SIZE = 32
    MAX_EPOCHS = 1000
    PATIENCE = 5
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # AdamW originally from https://arxiv.org/abs/1711.05101 -- quite common now
    # note that the original 2017 paper used plain adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_validation_loss = float("inf")
    best_model_state_dict = None
    bad_trend_duration = 0

    for epoch in range(MAX_EPOCHS):
        epoch_train_loss = train_one_epoch(model, train_loader, optimizer)
        epoch_validation_loss = evaluate(model, val_loader)
        print(f"epoch {epoch + 1}: training loss {epoch_train_loss:.4f},  validation loss {epoch_validation_loss:.4f}")
        results["Epoch_number"].append(epoch + 1)
        results["training_loss"].append(epoch_train_loss)
        results["validation_loss"].append(epoch_validation_loss)

        if epoch_validation_loss > best_validation_loss:
            bad_trend_duration += 1
        else:
            bad_trend_duration = 0
            best_validation_loss = epoch_validation_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

        if bad_trend_duration > PATIENCE:
            print(f"Stopping after {epoch + 1} epochs.")
            break

    model.load_state_dict(best_model_state_dict)
    test_loss = evaluate(model, test_loader)
    print(f"test loss: {test_loss:.4f}")
    torch.save(model.state_dict(), f"transformer_weights_for_classifier_{time.time()}.pt")
    

# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# https://docs.pytorch.org/docs/2.15/generated/torch.nn.Unfold.html
def train_one_epoch(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer):
    model.train() # why is the mode toggle implemented as a random function call? who knows
    total_pertoken_loss = 0.0

    for i, data in enumerate(data_loader):
        x, x_shifted, y, length = (t.to("cuda") for t in data) # figure out a better way to keep things tied to device
        
        optimizer.zero_grad()
        logits = model(x)

        # we've both batch and sequence dimensions first, we flatten them together for cross_entropy requirements
        logits_flattened = logits.reshape(-1, VOCABULARY_SIZE)
        target_flattened = x_shifted.reshape(-1)
        loss = F.cross_entropy(logits_flattened, target_flattened, ignore_index=IGNORE_INDEX, reduction="mean")
        loss.backward()
        optimizer.step()

        total_pertoken_loss += loss.item()

    # naming is a bit confusing here, since it's already reduced by mean in cross entropy
    # we will be returning our actual average single amino acid / token prediction loss per batch, averaged over the batches
    # actually nvm words are hard, rewrite these comments later
    average_pertoken_loss = total_pertoken_loss / len(data_loader)
    return average_pertoken_loss


# https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html
# https://www.geeksforgeeks.org/python/what-is-with-torch-no_grad-in-pytorch/
@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader):
    model.eval() # :)
    total_eval_loss_sum = 0.0
    total_eval_token_count = 0

    for i, data in enumerate(data_loader):
            x, x_shifted, y, length = (t.to("cuda") for t in data)
    
            logits = model(x)
            
            # we've both batch and sequence dimensions first, we flatten them together for cross_entropy requirements
            logits_flattened = logits.reshape(-1, VOCABULARY_SIZE)
            target_flattened = x_shifted.reshape(-1)
            loss = F.cross_entropy(logits_flattened, target_flattened, ignore_index=IGNORE_INDEX, reduction="sum")
    
            total_eval_loss_sum += loss.item()
            total_eval_token_count += (target_flattened != IGNORE_INDEX).sum().item() # gpu single-instruction magic

    average_eval_token_loss = total_eval_loss_sum / total_eval_token_count
    return average_eval_token_loss


if __name__ == "__main__":
    main()

    #Plotting loss curves
import matplotlib.pyplot as plt

Learning_rate = LEARNING_RATE
optimizer = "Adam"
batch_size = 32

plt.plot(results["Epoch_number"], results["training_loss"], label="Training Loss")
plt.plot(results["Epoch_number"], results["validation_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# hyperparameters as a separate legend entry
plt.plot([], [], " ", label=f"Optimizer: {optimizer}")
plt.plot([], [], " ", label=f"Learning rate: {Learning_rate}")
plt.plot([], [], " ", label=f"Batch size: {batch_size}")

plt.title("Training and Validation Loss Curves for Transformer Model")
plt.legend()
plt.show()

"""
epoch 1: training loss 2.1090,  validation loss 1.9847
epoch 2: training loss 1.6919,  validation loss 1.8510
epoch 3: training loss 1.5592,  validation loss 1.7545
epoch 4: training loss 1.4776,  validation loss 1.6957
epoch 5: training loss 1.4186,  validation loss 1.6702
epoch 6: training loss 1.3702,  validation loss 1.6631
epoch 7: training loss 1.3308,  validation loss 1.6357
epoch 8: training loss 1.2971,  validation loss 1.6189
epoch 9: training loss 1.2655,  validation loss 1.5827
epoch 10: training loss 1.2381,  validation loss 1.5794
epoch 11: training loss 1.2121,  validation loss 1.5892
epoch 12: training loss 1.1902,  validation loss 1.5455
epoch 13: training loss 1.1686,  validation loss 1.5829
epoch 14: training loss 1.1473,  validation loss 1.5058
epoch 15: training loss 1.1294,  validation loss 1.5270
epoch 16: training loss 1.1123,  validation loss 1.5412
epoch 17: training loss 1.0947,  validation loss 1.5223
epoch 18: training loss 1.0792,  validation loss 1.5295
epoch 19: training loss 1.0631,  validation loss 1.5263
epoch 20: training loss 1.0497,  validation loss 1.5240
Stopping after 20 epochs.
test loss: 1.5286
"""