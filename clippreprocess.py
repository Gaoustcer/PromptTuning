from dataset import dataset
# for image_class
import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model,preprocess = clip.load("ViT-B/32",device)
image_embedding = []
label_embedding = []
from tqdm import tqdm
for imgs,labels in tqdm(dataset):
    image_embedding.append(model.encode_image(
        preprocess(imgs).unsqueeze(0).to(device)
    ))
    label_embedding.append(labels)
image_embedding = torch.stack(image_embedding,dim=0)
label_embedding = torch.LongTensor(label_embedding)
torch.save(image_embedding,'dataset/images.pth')
torch.save(label_embedding,"dataset/labels.pth")
