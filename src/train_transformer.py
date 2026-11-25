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

LEARNING_RATE = 0.0005

def main():
    model = Transformer().to("cuda")

    train_data = ProteinDataset(path_to_data="data/train.csv")
    val_data = ProteinDataset(path_to_data="data/val.csv")
    test_data = ProteinDataset(path_to_data="data/test.csv")

    BATCH_SIZE = 32
    MAX_EPOCHS = 1000
    PATIENCE = 10
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # AdamW originally from https://arxiv.org/abs/1711.05101 -- quite common now
    # note that the original 2017 paper used plain adam
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay = 0.1, lr=LEARNING_RATE)

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
    torch.save(model.state_dict(), f"medium_transformer_weights_for_classifier_{time.time()}.pt")
    

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
optimizer = "AdamW"
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
transformer 512

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


transformer 256
epoch 1: training loss 2.2481,  validation loss 2.0493
epoch 2: training loss 1.8868,  validation loss 1.9447
epoch 3: training loss 1.7605,  validation loss 1.8867
epoch 4: training loss 1.6804,  validation loss 1.8760
epoch 5: training loss 1.6217,  validation loss 1.7845
epoch 6: training loss 1.5747,  validation loss 1.7948
epoch 7: training loss 1.5384,  validation loss 1.7547
epoch 8: training loss 1.5065,  validation loss 1.7338
epoch 9: training loss 1.4801,  validation loss 1.6857
epoch 10: training loss 1.4585,  validation loss 1.6833
epoch 11: training loss 1.4373,  validation loss 1.6891
epoch 12: training loss 1.4184,  validation loss 1.6422
epoch 13: training loss 1.4019,  validation loss 1.6620
epoch 14: training loss 1.3874,  validation loss 1.6175
epoch 15: training loss 1.3749,  validation loss 1.6336
epoch 16: training loss 1.3616,  validation loss 1.6361
epoch 17: training loss 1.3499,  validation loss 1.5960
epoch 18: training loss 1.3405,  validation loss 1.6125
epoch 19: training loss 1.3305,  validation loss 1.5977
epoch 20: training loss 1.3218,  validation loss 1.5780
epoch 21: training loss 1.3130,  validation loss 1.5694
epoch 22: training loss 1.3057,  validation loss 1.6287
epoch 23: training loss 1.2969,  validation loss 1.5806
epoch 24: training loss 1.2903,  validation loss 1.5801
epoch 25: training loss 1.2844,  validation loss 1.6021
epoch 26: training loss 1.2771,  validation loss 1.6040
epoch 27: training loss 1.2722,  validation loss 1.5686
epoch 28: training loss 1.2657,  validation loss 1.5329
epoch 29: training loss 1.2614,  validation loss 1.5784
epoch 30: training loss 1.2563,  validation loss 1.5552
epoch 31: training loss 1.2503,  validation loss 1.5628
epoch 32: training loss 1.2458,  validation loss 1.5429
epoch 33: training loss 1.2409,  validation loss 1.5524
epoch 34: training loss 1.2367,  validation loss 1.5725
epoch 35: training loss 1.2318,  validation loss 1.5296
epoch 36: training loss 1.2291,  validation loss 1.5335
epoch 37: training loss 1.2241,  validation loss 1.5376
epoch 38: training loss 1.2194,  validation loss 1.5474
epoch 39: training loss 1.2164,  validation loss 1.5611
epoch 40: training loss 1.2132,  validation loss 1.5302
epoch 41: training loss 1.2091,  validation loss 1.5265
epoch 42: training loss 1.2056,  validation loss 1.5417
epoch 43: training loss 1.2023,  validation loss 1.5011
epoch 44: training loss 1.1992,  validation loss 1.5242
epoch 45: training loss 1.1971,  validation loss 1.5230
epoch 46: training loss 1.1937,  validation loss 1.5301
epoch 47: training loss 1.1909,  validation loss 1.5273
epoch 48: training loss 1.1881,  validation loss 1.5256
epoch 49: training loss 1.1849,  validation loss 1.4993
epoch 50: training loss 1.1820,  validation loss 1.5153
epoch 51: training loss 1.1797,  validation loss 1.5401
epoch 52: training loss 1.1778,  validation loss 1.4950
epoch 53: training loss 1.1758,  validation loss 1.5209
epoch 54: training loss 1.1731,  validation loss 1.5306
epoch 55: training loss 1.1706,  validation loss 1.5655
epoch 56: training loss 1.1679,  validation loss 1.5009
epoch 57: training loss 1.1658,  validation loss 1.5071
epoch 58: training loss 1.1644,  validation loss 1.4948
epoch 59: training loss 1.1618,  validation loss 1.5147
epoch 60: training loss 1.1595,  validation loss 1.5142
epoch 61: training loss 1.1594,  validation loss 1.5098
epoch 62: training loss 1.1568,  validation loss 1.5029
epoch 63: training loss 1.1545,  validation loss 1.5194
epoch 64: training loss 1.1536,  validation loss 1.5073
epoch 65: training loss 1.1513,  validation loss 1.4982
epoch 66: training loss 1.1497,  validation loss 1.4982
epoch 67: training loss 1.1477,  validation loss 1.4901
epoch 68: training loss 1.1467,  validation loss 1.4800
epoch 69: training loss 1.1449,  validation loss 1.5066
epoch 70: training loss 1.1428,  validation loss 1.5128
epoch 71: training loss 1.1417,  validation loss 1.5069
epoch 72: training loss 1.1409,  validation loss 1.5052
epoch 73: training loss 1.1387,  validation loss 1.4816
epoch 74: training loss 1.1368,  validation loss 1.4945
epoch 75: training loss 1.1366,  validation loss 1.4877
epoch 76: training loss 1.1352,  validation loss 1.4937
epoch 77: training loss 1.1330,  validation loss 1.4878
epoch 78: training loss 1.1320,  validation loss 1.4766
epoch 79: training loss 1.1315,  validation loss 1.4830
epoch 80: training loss 1.1302,  validation loss 1.4834
epoch 81: training loss 1.1281,  validation loss 1.4821
epoch 82: training loss 1.1270,  validation loss 1.5048
epoch 83: training loss 1.1262,  validation loss 1.4696
epoch 84: training loss 1.1249,  validation loss 1.5014
epoch 85: training loss 1.1238,  validation loss 1.5138
epoch 86: training loss 1.1222,  validation loss 1.5090
epoch 87: training loss 1.1208,  validation loss 1.4748
epoch 88: training loss 1.1210,  validation loss 1.4912
epoch 89: training loss 1.1192,  validation loss 1.4789
epoch 90: training loss 1.1191,  validation loss 1.4941
epoch 91: training loss 1.1172,  validation loss 1.4735
epoch 92: training loss 1.1164,  validation loss 1.4826
epoch 93: training loss 1.1150,  validation loss 1.4942
epoch 94: training loss 1.1144,  validation loss 1.5007
Stopping after 94 epochs.
test loss: 1.4785
"""