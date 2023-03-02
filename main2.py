import copy
import os.path
import time
from torchvision import models
import torch
import torchvision
from customdata import my_customdata
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd
from imblearn.over_sampling import SMOTE #불균형 데이터 문제를 해결 pip install -U imbalanced-learn
import numpy as np
from torchvision import datasets

import os
device = torch.device("cuda")#if torch.cuda.is_available() else "cpu")




train_transforms = A.Compose([
    #A.Resize(width=224, height=224, always_apply=True),
    A.SmallestMaxSize(max_size=224),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=20,
                       p=0.8),
    A.RandomShadow(p=.6),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224,0.255)),
    ToTensorV2(),
])

val_transforms = A.Compose([
    #A.resize(width=224, height=224, always_apply=True),
    A.SmallestMaxSize(max_size=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)),
    ToTensorV2()
])

# dataset
train_dataset = my_customdata("./dataset/train/", transform=train_transforms)
val_dataset = my_customdata("./dataset/val/", transform=val_transforms)
test_dataset = my_customdata("./dataset/test/", transform=val_transforms)

# dataloader


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True )#,num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)#,num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 아래는 트랜스비전 모델
# model = torch.hub.load('facebookresearch/deit:main',
#                        'deit_tiny_patch16_224', pretrained=False)
# model.head = nn.Linear(in_features=192, out_features=5) #애는 출력층이 헤드임 왠진 몰루
# model.to(device)


# model = models.efficientnet_b4(pretrained=True)
# model.classifier[1] = torch.nn.Linear(in_features=1792,out_features=5)#450개로 분류하잖음
# model.to(device)


model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(in_features=2048, out_features=10)
model.to(device)

# resnet18 모델 방금 이거씀
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(in_features=512, out_features=10)  # 450개로 분류하잖음
# model.to(device)

# model = models.squeezenet1_1(pretrained=True)
# model.fc = nn.Linear(in_features=512, out_features=5)  # 450개로 분류하잖음
# model.to(device)

# model = models.VisionTransformer(image_size=224,patch_size=224,num_layers=6,num_heads=5, hidden_dim= 5, mlp_dim= 3)
# model.head = nn.Linear(in_features=1000, out_features=5)
# model.to(device)


# model = models.vit_l_32()
# model.head = nn.Linear(in_features=1024, out_features=5)
# model.to(device)

#모델 불러오기
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(in_features=512, out_features=10) #불러올때 특징까지 다 다시 넣어줘야함
# model = torch.load("./best.pt")
# model.eval()

#가중치 함수
num_ins = ['age20_f', 'age20_m', 'age30_f', 'age30_m', 'age40_f', 'age40_m', 'age50_f', 'age50_m', 'age60_f', 'age60_m']  # 실제 클래스 수
num_ins = [30476, 47990, 9988, 22636, 382, 1264, 72, 92, 54, 68]
weights = [1 - (x / sum(num_ins)) for x in num_ins]
class_weights = torch.FloatTensor(weights).to(device)


criterion = nn.CrossEntropyLoss(weight=class_weights)#LabelSmoothingCrossEntropy()#   #여기서 weight값을 넣으면 가중치가 들어감
#optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.005) #이거는 위에 다른 모델에서 씀
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) #이거는 레즈넷에서 씀

# lr scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30,
                                                   gamma=0.1)

# sm = SMOTE(sampling_strategy='auto', random_state=0)
#
# x_smote, y_smote = sm.fit_resample(x_train,y_train) #오버샘플링 코드지만 아직 이해불가능하므로 일단 내버려둠

num_epochs = 0

def train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs=2,
          device=device) :
    total = 0
    best_loss = 9999
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    model.train()
    for epoch in range(num_epochs) :
        print(f"Epoch {epoch} / {num_epochs - 1}")
        print("-"*10)

        for index, (image, label) in enumerate(train_loader) :
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            # print(f"model={model}")
            # print(f"output ={output}")
            # print(f"label = {label}")
            # print(f"loss = {loss}")
            # print(f"output ={output.shape}")
            # print(f"label = {label.shape}")
            # print(f"loss = {loss.shape}")
            # exit()
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1) #argmax는 쉽게 말해서 테스트가 예측한 라벨값이다
            # torch.max를 하면 output으로 나온값들중 제일큰 것의 값을 알려주고 그것의 리스트번호 (예를들어서 아웃풋이 5개가 나와서 1 2 3 5 4 이라고 칠때, 가장큰 값은 5 가장 큰 값의 리스트(라벨번호)는 3이다.
            # 즉 어규맥스에 가장 큰 값의 리스트번호인 3이 들어간다는 것이다. 이것이 테스트가 예측한 라벨값이다.
            acc = (label == argmax).float().mean() #여기서 그것을 비교해준다.
            total += label.size(0)
            # temp = label.size(0) # 지금까지 처리한 데이터의 양을 말해줌 label.size(0)에서 size(0)은 차원관련 내용
            # print(f"temp = {temp}")
            # exit()

            if (index + 1) % 10 == 0 :
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, Acc {:.2f}".format(
                    epoch + 1, num_epochs, index+1, len(train_loader), loss.item(),
                    acc.item() * 100
                ))
        aveg_loss, val_acc = validation(epoch, model, val_loader, criterion, device, num_epochs)
        if aveg_loss < best_loss :
            best_loss = aveg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(model, save_dir="./")

    time_elapsed = time.time()  - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 6
    ))


def validation(epoch, model, val_loader, criterion, device, num_epochs) :
    print("Start validation # {}" .format(epoch+1))

    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(val_loader) :
            imags, labels = imgs.to(device), labels.to(device)
            output = model(imags)
            loss = criterion(output, labels)
            batch_loss += loss.item()

            total += imags.size(0)
            _, argmax = torch.max(output, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt +=1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("Validation #{} Acc : {:.2f}% Average Loss : {:.4f}".format(
        epoch + 1,
        correct / total * 100,
        avrg_loss
    ))

    dfForAccuracy = pd.DataFrame(index=list(range(epoch)),
                                 columns=['Epoch', 'Accuracy','AVG_loss'])

    dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
    dfForAccuracy.loc[epoch, 'Accuracy'] = val_acc.item()
    dfForAccuracy.loc[epoch, 'AVG_loss'] = avrg_loss.item()

    #if num_epochs == epoch - 1:
    dfForAccuracy.to_csv("./modelAccuracy.csv", index=False)


    return avrg_loss, val_acc


def save_model(model, save_dir, file_name = "best.pt") :
    output_path = os.path.join(save_dir,file_name)
    torch.save(model.state_dict(), output_path)

#테스트 부분
def test(model, test_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i ,(image,labels) in enumerate(test_loader):
            image, labels = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output,1)
            total += image.size(0)
            correct += (labels == argmax).sum().item()
        acc = correct / total * 100
        print("acc for {} image: {:.2f}%".format(total, acc))


if __name__ == "__main__" :

    #모델을 테스트함
    model.load_state_dict(torch.load("./resnet50.pt", map_location=device))
    test(model, test_loader, device)

    # #트레인함 모델 테스트할때는 아래 주석으로해야함
    # train(model, criterion,train_loader, val_loader, optimizer,
    #       scheduler=exp_lr_scheduler, device=device)
