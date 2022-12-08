import torch
import tqdm
from src.data.loss import DetectorLoss
from superpoint import superpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = DetectorLoss(8)
optimizer = torch.optim.Adam(superpoint.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.7, threshold=0.01, verbose=True)


def train(model, dataloader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss        = 0.0
    
    for i, (img, pts) in enumerate(dataloader):

        optimizer.zero_grad()

        img, pts = img.to(DEVICE), pts.to(DEVICE)

        logits = model(img)

        loss = criterion(logits, pts) #[None, :, :])

        running_loss        += loss.item()

        loss.backward()
        optimizer.step()

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()

        del img, pts
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)

    return running_loss


def validate(model, dataloader):

    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    val_loss = 0.0

    for i, (img, pts) in enumerate(dataloader):

        img, pts = img.to(DEVICE), pts.to(DEVICE)

        with torch.inference_mode():
            logits = model(img)

        loss = criterion(logits, pts) #[None, :, :])

        val_loss        += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(val_loss/(i+1)))
        batch_bar.update()

        del img, pts
        torch.cuda.empty_cache()

    batch_bar.close()
    val_loss /= len(dataloader)
    scheduler.step(val_loss)

    return val_loss


def testing(model, dataloader):
    model.eval()
    total_loss = 0
    
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    for i, (img, pts) in enumerate(dataloader):

        img, pts = img.to(DEVICE), pts.to(DEVICE)

        with torch.no_grad():
            prediction = model(img)
            loss = criterion(prediction, pts)
            batch_size = img.shape[0]
            total_loss += loss.item() * batch_size
        break
        batch_bar.update()
    return total_loss / len(dataloader)