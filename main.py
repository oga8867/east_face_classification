import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from customdata import my_customdata
from torchvision import models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.is_available())
# exit()
# 1 transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(25),
    # transforms.RandomEqualize(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
# 2 data set data loader
train_dataset = my_customdata("./dataset/train/", transform=train_transforms)
val_dataset = my_customdata("./dataset/val/", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3 model call

# net = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)
# # net = torchvision.models.vit_b_16()
# # print(net)
# # exit()
# net.head = torch.nn.Linear(in_features=768, out_features=5)
# # print(net)
# # exit()
#
# # net = models.__dict__["resnet50"](pretrained=False)
# # net.fc = torch.nn.Linear(in_features=2048, out_features=5)
# net.to(device)


net = models.resnet18(pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=5)  # 450개로 분류하잖음
net.to(device)
# 4 train loop
train_losses = []
val_losses = []
train_accs = []
val_accs = []


def train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
          device):
    print("train ....!!! ")
    total = 0
    best_loss = 7777

    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            img, labels = images.to(device), labels.to(device)

            # model <- img
            output = model(img)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}] Step [{}/{}] Loss {:.4f} Acc {:.2f}".format(
                    epoch + 1,
                    num_epoch,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                    acc.item() * 100
                ))
                print(f"argmax[:10] = {argmax[:10]}, label = {labels[:10]}")
        train_losses.append(loss.item())
        train_accs.append(acc.item() * 100)
        avrg_loss, val_acc = validation(epoch, model, val_loader, criterion, device)
        val_losses.append(avrg_loss.item())
        val_accs.append(val_acc)
        if avrg_loss < best_loss:
            print("Best acc save !!! ")
            best_loss = avrg_loss
            torch.save(model.state_dict(), "./best_vit.pt")
        print(f"train_loss = {train_losses}, train_acc = {train_accs}")

        print(f"val_loss = {val_losses}, val_acc = {val_accs}")
    torch.save(model.state_dict(), "./last_vit.pt")


# 5. val loop
def validation(epoch, model, val_loader, criterion, device):
    print(f"validation .... {epoch} ")
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        for i, (images, labels) in enumerate(val_loader):
            img, label = images.to(device), labels.to(device)

            # model <- img
            output = model(img)
            loss = criterion(output, label)
            batch_loss += loss.item()

            total += img.size(0)
            _, argmax = torch.max(output, 1)
            correct += (label == argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("acc : {:.2f}% loss : {:.4f}".format(
        val_acc, avrg_loss
    ))
    model.train()
    return avrg_loss, val_acc


# 0 Hyper parameter
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)


def test(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (image, labels) in enumerate(val_loader):
            image, label = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)

            total += image.size(0)
            print(image.size(0), total)  # 128
            correct += (label == argmax).sum().item()

        acc = correct / total * 100
        print("acc for {} image : {:.2f}%".format(
            total, acc
        ))


if __name__ == "__main__":  # main
    # net.load_state_dict(torch.load("./best.pt", map_location=device))
    # test(net, val_loader, device)
    epoch = 5
    train(num_epoch=epoch, model=net, train_loader=train_loader, val_loader=val_loader,
          criterion=criterion, optimizer=optimizer, device=device)
    plt.subplot(1, 2, 1)

    plt.title("Training and Validation Loss")
    plt.plot(np.arange(1, epoch + 1), val_losses, label="val")
    plt.plot(np.arange(1, epoch + 1), train_losses, label="train")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    #    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(np.arange(1, epoch + 1), val_accs, label="val")
    plt.plot(np.arange(1, epoch + 1), train_accs, label="train")
    plt.xlabel("epochs")
    plt.ylabel("acuuracy")
    plt.legend()

    # plt.show()
    plt.savefig("myfig_vit.png")
    # fig.show()