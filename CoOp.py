from dataset import model,classes
from torch.utils.data import Dataset,DataLoader
import torch
from torchmetrics import Accuracy
import torch.nn.functional as F
import torch.nn as nn
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
accuracy = Accuracy("multiclass",num_classes=len(classes),top_k=1).to(device)
def accuracybatch(text_embedding,image_embedding,label):
    text_embedding = F.normalize(text_embedding,dim = -1)
    image_embedding = F.normalize(image_embedding,dim = -1)
    score = image_embedding @ text_embedding.t()
    return accuracy(score,label)

class Clipdataset(Dataset):
    def __init__(self):
        self.imgs = torch.load("./dataset/imgs.pt",map_location=device)
        self.labels = torch.load("./dataset/labels.pt",map_location=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return self.imgs[idx],self.labels[idx]

class PromptLearning(nn.Module):
    def __init__(self,ctx_init = "this is ",classes = classes,n_ctx = 16):
        super().__init__()
        n_cls = len(classes)
        dtype = model.dtype
        ctx_dim = model.ln_final.weight.shape[0]
        prompt = clip.tokenize(ctx_init).to(device)
        embedding = model.token_embedding(prompt).detach()
        optimized_ctx = embedding[0,1:1 + n_ctx,:]
        self.optimized_ctx = nn.Parameter(optimized_ctx)
        classnames = []
        for cname in classes:
            classnames.append(cname.replace("_"," "))
        PROMPT = [ctx_init + name + "." for name in classnames]
        tokenrompt = torch.cat([clip.tokenize(p) for p in PROMPT])
        self.tokenprompt = tokenrompt
        embedding = model.token_embedding(tokenrompt).to(device)
        self.token_prefix = embedding[:,:1,:]
        self.token_suffix = embedding[:,1 + n_ctx:,:]
        self.n_cls = n_cls

    def forward(self):
        ctx = self.optimized_ctx.unsqueeze(0).repeat(self.n_cls,1,1)
        # print(self.token_prefix.shape,self.token_suffix.shape,ctx.shape)
        return torch.concat(
            [self.token_prefix,ctx,self.token_suffix],dim=1
        )
        # Input to the transformer should be [batchsize,sequencelen,ctx_dim]
if __name__ == "__main__":
    tokenprompt = PromptLearning()
    print(tokenprompt().shape)
    # pass
