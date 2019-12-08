#!/usr/bin/env python
# coding: utf-8

# In[1]:


#General Imports
import torch
import torch.nn  as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import random

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

#Load fake, non handwritten generator 
from fake_texts.pytorch_dataset_fake_2 import Dataset

#Import the loss from baidu 
from torch.nn import CTCLoss

#Import the model 
from fully_conv_model import cnn_attention_ocr

#Helper to count params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Evaluation function preds_to_integer
from evaluation import wer_eval,preds_to_integer,show,my_collate,AverageMeter


# In[2]:


#Set up Tensorboard writer for current test
writer = SummaryWriter(log_dir="/home/leander/AI/repos/OCR-CNN/logs2/correct_cosine_2")


# In[3]:


###Set up model. 
cnn=cnn_attention_ocr(model_dim=64,nclasses=67,n_layers=8)
cnn=cnn.cuda().train()
count_parameters(cnn)


# In[4]:


#cnn=cnn.eval()


# In[5]:


#CTC Loss ,average_frames=True
ctc_loss = CTCLoss(reduction="mean")
#Optimizer: Good initial is 5e5 
optimizer = optim.Adam(cnn.parameters(), lr=5e-4)


# In[6]:


#We keep track of the Average loss and CER 
ave_total_loss = AverageMeter()
CER_total= AverageMeter()


# In[7]:


n_iter=0
batch_size=4


# In[8]:


#torch.save(cnn.state_dict(), "400ksteps_augment_new_gen_e15.pt")


# In[9]:


torch.save(cnn.state_dict(), "415ksteps_augment_new_gen_e56.pt")


# In[8]:


#
cnn.load_state_dict(torch.load("415ksteps_augment_new_gen_e56.pt"))


# In[8]:


ds=Dataset()


# In[9]:


trainset = DataLoader(dataset=ds,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=6,
                      pin_memory=True,
                  
                      collate_fn=my_collate)
gen = iter(trainset)


# In[10]:


from torch.optim.lr_scheduler import CosineAnnealingLR


# In[11]:


cs=CosineAnnealingLR(optimizer=optimizer,T_max=250000,eta_min=1e-6)


# In[12]:


npa=1

for epochs in range(10000):

    gen = iter(trainset)
    print("train start")
    for i,ge in enumerate(gen):
        
        #to avoid OOM 
        if ge[0].shape[3]<=800:

            #DONT FORGET THE ZERO GRAD!!!!
            optimizer.zero_grad()
            
            #Get Predictions, permuted for CTC loss 
            log_probs = cnn(ge[0].cuda()).permute((2,0,1))

            #Targets have to be CPU for baidu loss 
            targets = ge[1]#.cpu()

            #Get the Lengths/2 becase this is how much we downsample the width
            input_lengths = ge[2]/2
            target_lengths = ge[3]
            
            #Get the CTC Loss 
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            #Then backward and step 
            loss.backward()
            optimizer.step()
            
            #Save Loss in averagemeter and write to tensorboard 
            ave_total_loss.update(loss.data.item())
            writer.add_scalar("total_loss", ave_total_loss.average(), n_iter) 
            
            
            #Here we Calculate the Character error rate
            cum_len=np.cumsum(target_lengths)
            targets=np.split(ge[1].cpu(),cum_len[:-1])
            wer_list=[]
            for j in range(log_probs.shape[1]):
                wer_list.append(wer_eval(log_probs[:,j,:][0:input_lengths[j],:],targets[j]))
            
            #Here we save an example together with its decoding and truth
            #Only if it is positive 
            
            if np.average(wer_list)>0.1:

                max_elem=np.argmax(wer_list)
                max_value=np.max(wer_list)
                max_image=ge[0][max_elem].cpu()
                max_target=targets[max_elem]
                
                max_target=[ds.decode_dict[x] for x in max_target.tolist()]
                max_target="".join(max_target)

                ou=preds_to_integer(log_probs[:,max_elem,:])
                max_preds=[ds.decode_dict[x] for x in ou]
                max_preds="".join(max_preds)
                
                writer.add_text("label",max_target,n_iter)
                writer.add_text("pred",max_preds,n_iter)
                writer.add_image("img",ge[0][max_elem].detach().cpu().numpy(),n_iter)
                
                #gen.close()
                #break
                
            #Might become infinite 
            if np.average(wer_list)< 10: 
                CER_total.update(np.average(wer_list))
                writer.add_scalar("CER", CER_total.average(), n_iter)
            
            #We save when the new avereage CR is beloew the NPA 
            if npa>CER_total.average() and CER_total.average()>0 and CER_total.average()<1:
                
                torch.save(cnn.state_dict(), "autosave.pt")
                npa=CER_total.average()
                
                
            n_iter=n_iter+1
            cs.step()
            lr=optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr",lr,n_iter)
            


# In[14]:


CER_total.average()


# In[ ]:


save_checkpoint({
    'epoch': epoch + 1,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
})


# In[15]:


cnn.load_state_dict(torch.load("autosave.pt"))


# In[16]:


optimizer.load_state_dict(torch.load("autosave_optimizer.pt"))


# In[ ]:




