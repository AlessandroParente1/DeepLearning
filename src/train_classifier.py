import copy
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from protgen.data.make_data import IGNORE_INDEX, VOCABULARY, VOCABULARY_SIZE, ProteinDataset

from protgen.models.clasifier_reuse import Transformer_Classifier
from protgen.models.transformer import Transformer

path_to_weights = "transformer_weights_for_classifier_1795441599.982232.pt"
LEARNING_RATE = 0.0003
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
    train_data = ProteinDataset(path_to_data="data/train.csv")
    val_data = ProteinDataset(path_to_data="data/val.csv")
    test_data = ProteinDataset(path_to_data="data/test.csv")

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


#This classifier will reuse masking used for generator training
# for example sample is:
#Position:  0    1    2    3    4    5    6    7 ...
#Token:    BOS   M    I    K    K    Q   EOS  PAD ...
#We need to know where to calculate loos - at what possitions 
# result tensor([False, True, True, True, True, True, False, False])
#valid positions are  to 1 to 5 
def create_valid_positions(lengths, sequence_length):


    #contains 0 to 128
    positions = torch.arange(sequence_length, device=lengths.device)

    valid_positions = (
        (positions.unsqueeze(0) >= 1) &
        (positions.unsqueeze(0) <= lengths.unsqueeze(1))
    )
    
    #Bool mask of where to calculate loos
    return valid_positions


# result tensor([False, True, True, True, True, True, False, False])
#instead of true i need the corect label
#first for every label instead of  0 = 1 i need 0 = 1,1,1,1,1,...to seq_lenght
#targets: family labels repeated across all positions.
# at the end targets to valid positions gives:
#Targets after:  [-1, 1, 1, 1, 1, 1, -1, -1, -1.....]
def create_targets(y, sequence_length, valid_positions):
    targets = y.unsqueeze(1).expand(-1, sequence_length).clone()
    #now lets undo bool and change true to label:
    targets[~valid_positions] = IGNORE_INDEX
    return targets





def train_one_epoch(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer):

    #ignore_index=-1: Ignore BOS, EOS and padding positions, whose targets we set to -1.
    #reduction="sum": Add the loss from all valid prefixes
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")

    model.train() # why is the mode toggle implemented as a random function call? who knows
    model.embedding_dropout.eval() #disables embedding dropout
    model.blocks.eval() #disables dropout in transformer blocks including attention dropout

    total_loss = 0.0
    total_valid_positions = 0

    for i, data in enumerate(data_loader):

        x, _, y, length = data 
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)

        #step1
        optimizer.zero_grad()
        #step2
        class_logits = model(x)
        #step3
        valid_positions = create_valid_positions(length, x.shape[1])
        targets = create_targets(y, x.shape[1], valid_positions)

        num_classes = class_logits.shape[-1]

        loss_sum = criterion(

            #class logits returns: [batch_size, sequence_length, num_classes]
            #but cross entropy expects: [number_of_predictions, number_of_classes]
            #-1 figure out first dimension automatically 32*129 = 4128 prediction for 3 classes
            class_logits.reshape(-1, num_classes),
            #gives target label for each prediction
            targets.reshape(-1)
        )

        #loss_sum / length.sum(): Calculate the average loss per valid prefix
        loss = loss_sum / length.sum()

        #step4
        loss.backward()
        #step5
        optimizer.step()

        total_loss += loss_sum.item()
        total_valid_positions += length.sum().item()

    average_loss = total_loss / total_valid_positions
    return average_loss


# https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html
# https://www.geeksforgeeks.org/python/what-is-with-torch-no_grad-in-pytorch/
@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader):

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")

    model.eval() # :)
    total_eval_loss_sum = 0.0
    total_valid_positions = 0

    for i, data in enumerate(data_loader):

        x, x_shifted, y, length = data 
        x = x.to(device)
        y = y.to(device)
        length = length.to(device)


        class_logits = model(x)
            
        # loss of one batch
        
        valid_positions = create_valid_positions(length, x.shape[1])
        targets = create_targets(y, x.shape[1], valid_positions)

        num_classes = class_logits.shape[-1]

        loss_sum = criterion(
            #class logits returns: [batch_size, sequence_length, num_classes]
            #but cross entropy expects: [number_of_predictions, number_of_classes]
            #-1 figure out first dimension automatically 32*129 = 4128 prediction for 3 classes
            class_logits.reshape(-1, num_classes),
            #gives target label for each prediction
            targets.reshape(-1)
        )

        #total_eval_loss_sum: summed loss accumulated across all batches.
        total_eval_loss_sum += loss_sum.item()

        #total_valid_positions: total number of valid prefixes across all batches.
        total_valid_positions += length.sum().item()

    #average_eval_loss: average loss per prefix across the entire evaluation dataset divided by the amount of valid positions in dataset
    average_eval_loss = total_eval_loss_sum / total_valid_positions
    return average_eval_loss


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

plt.title("Training and Validation Loss Curves for Clasifier Model")
plt.legend()
plt.show()


"""
epoch 1: training loss 0.2132,  validation loss 0.1158
epoch 2: training loss 0.1051,  validation loss 0.1041
epoch 3: training loss 0.0920,  validation loss 0.1001
epoch 4: training loss 0.0861,  validation loss 0.0978
epoch 5: training loss 0.0820,  validation loss 0.0967
epoch 6: training loss 0.0797,  validation loss 0.0993
epoch 7: training loss 0.0778,  validation loss 0.0962
epoch 8: training loss 0.0762,  validation loss 0.0957
epoch 9: training loss 0.0744,  validation loss 0.0964
epoch 10: training loss 0.0732,  validation loss 0.0962
epoch 11: training loss 0.0721,  validation loss 0.0956
epoch 12: training loss 0.0708,  validation loss 0.0940
epoch 13: training loss 0.0700,  validation loss 0.0961
epoch 14: training loss 0.0689,  validation loss 0.0954
epoch 15: training loss 0.0683,  validation loss 0.0959
epoch 16: training loss 0.0675,  validation loss 0.0990
epoch 17: training loss 0.0666,  validation loss 0.0957
Stopping after 17 epochs.
test loss: 0.1203
"""



