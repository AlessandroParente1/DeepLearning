import copy
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from protgen.data.make_data_classifier import VOCABULARY, VOCABULARY_SIZE, ProteinPrefixDataset

from protgen.models.clasifier_reuse import Transformer_Classifier
from protgen.models.transformer import Transformer

path_to_weights = "transformer_weights_for_classifier_1795441599.982232.pt"
LEARNING_RATE = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for plotting loss curves 
results = {
    "Epoch_number": [],
    "validation_loss": [],
    "training_loss": []
}


def main():

    #create instance of transformer
    generator = Transformer()

    #load the save weights
    generator.load_state_dict(torch.load(path_to_weights, map_location= device, weights_only =True))

    #Create instance of the clasifier
    model = Transformer_Classifier().to(device)

    #COPY PARAMS:

    #lets steal token_embedings:
    model.token_embedding.load_state_dict(generator.token_embedding.state_dict())

    #lets steal position embedings:
    model.position_embedding.load_state_dict(generator.position_embedding.state_dict())

    #lets steal steal learned parameters 
    model.blocks.load_state_dict(generator.blocks.state_dict())

    #normalization of the final hiden representations we have scale and boases from here.
    model.final_norm.load_state_dict(generator.final_norm.state_dict())

    #LETS FREEZE THE PARAMS:

    for param in model.parameters():
        param.requires_grad = False

    #Exept:
    for param in model.classifier_head.parameters():
        param.requires_grad = True

    # We still need to train te last layer - the head of the clasifier:

    #this returns x, x_shifted, label, lenght, we will not need x_shifted.
    train_data = ProteinPrefixDataset(path_to_data="data/train.csv")
    val_data = ProteinPrefixDataset(path_to_data="data/val.csv")
    test_data = ProteinPrefixDataset(path_to_data="data/test.csv")

    BATCH_SIZE = 32
    MAX_EPOCHS = 1000
    PATIENCE = 5
    #also data_loader returns x, labels, lengths needs unpacking!
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # AdamW originally from https://arxiv.org/abs/1711.05101 -- quite common now
    # note that the original 2017 paper used plain adam
    optimizer = torch.optim.Adam(model.classifier_head.parameters(), lr=LEARNING_RATE)
    

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

        if epoch_validation_loss < best_validation_loss:
            bad_trend_duration = 0
            best_validation_loss = epoch_validation_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

        else:
            bad_trend_duration += 1
            

        if bad_trend_duration >= PATIENCE:
            print(f"Stopping after {epoch + 1} epochs.")
            break

    #after all the training epochs finishes:
    if best_model_state_dict is None:
        raise RuntimeError("No best model checkpoint was saved.")

    model.load_state_dict(best_model_state_dict)
    test_loss = evaluate(model, test_loader)
    print(f"test loss: {test_loss:.4f}")
    #I do not really need the classifier weights for anything or do i?
    torch.save(model.state_dict(), f"classifier_weights_{time.time()}.pt")
    

# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# https://docs.pytorch.org/docs/2.15/generated/torch.nn.Unfold.html
def train_one_epoch(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer):
    
    criterion = nn.CrossEntropyLoss()

    model.train() # why is the mode toggle implemented as a random function call? who knows
    model.embedding_dropout.eval() #disables embedding dropout
    model.blocks.eval() #disables dropout in transformer blocks including attention dropout

    total_loss = 0.0

    for i, data in enumerate(data_loader):

        x, y, length = data 
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)

        #step1
        optimizer.zero_grad()
        #step2
        class_logits = model(x, length)
        #step3
        loss = criterion(class_logits, y)
        #step4
        loss.backward()
        #step5
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    # use sample-weighted average

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss


# https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html
# https://www.geeksforgeeks.org/python/what-is-with-torch-no_grad-in-pytorch/
@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader):

    criterion = nn.CrossEntropyLoss()

    model.eval() # :)
    total_eval_loss_sum = 0.0

    for i, data in enumerate(data_loader):

        x, y, length = data 
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)


        class_logits = model(x, length)
            
        # loss of one batch
        loss = criterion(class_logits, y)

        #multiply each batch's average loss by its number of samples:
        total_eval_loss_sum += loss.item() * x.size(0)

        
    average_eval_loss = total_eval_loss_sum / len(data_loader.dataset)
    return average_eval_loss


if __name__ == "__main__":
    main()

#   w   what is needed to do the current implementation trains on FULL LENGHT SEQUENCES
# IWANT TO TRAIN ON PREFIXES - SMALL PARTS OF THE FULL SEQUENCE 


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

plt.title("Training and Validation Loss Curves for Clasifier Model")
plt.legend()
plt.show()



