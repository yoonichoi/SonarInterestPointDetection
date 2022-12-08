import torch
from dataloader import SonarDataset

BATCH_SIZE = 16
root = '/content/sonar-data'

# Create objects for the dataset class
train_data = SonarDataset(root)
val_data = SonarDataset(root,"valid") 
test_data = SonarDataset(root,"test") 

train_loader = torch.utils.data.DataLoader(train_data, num_workers= 0,
                                           batch_size=BATCH_SIZE, 
                                           shuffle= True)
val_loader = torch.utils.data.DataLoader(val_data, num_workers= 0,
                                         batch_size=BATCH_SIZE, 
                                         shuffle= False)
test_loader = torch.utils.data.DataLoader(test_data, num_workers= 2, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle= False)

print("Batch size: ", BATCH_SIZE)
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))