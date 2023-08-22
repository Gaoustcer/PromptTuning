from dataset import model,preprocess,dataset
save_embeddings = []
labels = []
import torch
import os
from tqdm import tqdm
model.eval()
imgs = torch.load('./dataset/imgs.pt').cuda()
embedding = model.encode_image(imgs)
torch.save(embedding,"./dataset/embedding.pt")
# for img,label in tqdm(dataset):
#     save_embeddings.append(preprocess(img))
#     # save_embeddings.append(model.encode_image(preprocess(img).cuda().unsqueeze(0)).squeeze())
#     labels.append(label)
# labels = torch.tensor(labels)
# save_embeddings = torch.stack(save_embeddings,dim=0)
# rootdir = "./dataset/"
# torch.save(labels,os.path.join(rootdir,"labels.pt"))
# torch.save(save_embeddings,os.path.join(rootdir,"imgs.pt"))
