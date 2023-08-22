from dataset import model,classes
from tqdm import tqdm
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
    score = model.logit_scale.exp() * image_embedding @ text_embedding.t()
    return accuracy(score,label) * image_embedding.shape[0]

class Clipdataset(Dataset):
    def __init__(self):
        self.imgs = torch.load("./dataset/imgs.pt",map_location=device)
        self.labels = torch.load("./dataset/labels.pt",map_location=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return self.imgs[idx],self.labels[idx]
    

class SimplePromplearning(nn.Module):
    def __init__(self,ctx_init = 'This is a photo of',classes = classes,ctx_len = 16):
        super().__init__()
        self.classnames = [cname.replace("_"," ") for cname in classes]
        self.prompts = [ctx_init + " " + description for description in self.classnames]
        self.ctx_dim = model.ln_final.weight.shape[0]
        optimparam = torch.empty(len(classes),ctx_len,self.ctx_dim)
        self.optimparam = nn.Parameter(optimparam)
        self.paddingparam = torch.zeros(len(classes),model.context_length - ctx_len,self.ctx_dim)
        # clipresult = clip.tokenize(self.prompts)
        self.tokenprompt = clip.tokenize(self.prompts)
        self.tokenembedding = model.token_embedding(self.tokenprompt)


    def forward(self):
        simpleadd = torch.concat((self.optimparam,self.paddingparam),dim = 1)
        return self.tokenembedding + simpleadd 
                
class PromptLearning(nn.Module):
    def __init__(self,ctx_init = "",classes = classes,n_ctx = 16):
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
        PROMPT = [ctx_init + " " + name + "." for name in classnames]
        tokenrompt = torch.cat([clip.tokenize(p) for p in PROMPT])
        self.tokenprompt = tokenrompt.to(device)
        with torch.no_grad():
            embedding = model.token_embedding(self.tokenprompt)
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
# class CoOpSimple(object):
#     def __init__(self,EPOCH):
#         self.EPOCH = EPOCH
#         self.initctx = "this is a photo of "
#         self.classes = []
#         for cname in self.classes:
#             self.classes.append(cname.) 
        
class CoOp(object):
    def __init__(self,EPOCH) -> None:
        self.EPOCH = EPOCH
        dataset = Clipdataset()
        self.totallen = len(dataset)        
        self.dataloader = DataLoader(dataset,batch_size=32)
        self.promptlearner = SimplePromplearning().to(device)
        self.optim = torch.optim.Adam(self.promptlearner.parameters(),lr = 4e-3)
        self.transformer = model.transformer
        self.position_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection 
        # transformer output text withdim 512 and it serves as a one layer linear map
    

    def _generate_text_embedding(self):
        prompts = self.promptlearner() + self.position_embedding
        tokenize_prompts = self.promptlearner.tokenprompt  
        prompts = prompts.permute(1,0,2).to(torch.half)
        prompts = self.transformer(prompts)
        prompts = prompts.permute(1,0,2)
        prompts = self.ln_final(prompts)
        return prompts[torch.arange(prompts.shape[0]),tokenize_prompts.argmax(dim=-1)] @ self.text_projection

    def _validate(self):
        count = 0
        
        for imgs,labels in tqdm(self.dataloader):
            img_encoder = model.encode_image(imgs)
            text_encoder = self._generate_text_embedding()
            count += accuracybatch(text_encoder,img_encoder,labels)
        return count/self.totallen

    def learn(self):
        for epoch in range(self.EPOCH):
            print("acc in epoch",epoch,self._validate())
            for imgs,labels in tqdm(self.dataloader):
                self.optim.zero_grad()
                img_emb = model.encode_image(imgs)
                text_emb = self._generate_text_embedding()
                img_emb = F.normalize(img_emb,dim = -1)
                text_emb = F.normalize(text_emb,dim = -1)
                score = model.logit_scale.exp() * img_emb @ text_emb.t()
                loss = F.cross_entropy(score,labels)
                loss.backward()
                self.optim.step()
def raw_prompt():
    prompts = ["{}.".format(cname) for cname in classes]
    prompts = clip.tokenize(prompts).to(device)
    text_embedding = model.encode_text(prompts)
    print(text_embedding.shape)
    count  = 0
    dataset = Clipdataset()
    loader = DataLoader(dataset,batch_size = 32)
    for imgs,labels in tqdm(loader):
        img_emb = model.encode_image(imgs)
        count += accuracybatch(text_embedding,img_emb,labels)
    print("raw performance acc rate",count/10000)
if __name__ == "__main__":
    raw_prompt()
    coop = CoOp(128)
    coop.learn()
    # tokenprompt = PromptLearning()
    # for param in tokenprompt.parameters():
        # print(param.shape)
    # print(tokenprompt().shape)
    # print(tokenprompt.parameters())
    # pass
