import torch
import numpy as np
import matplotlib.pyplot as plt

    
def wer(r, h):
    '''
    Word error rate computation

    https://en.wikipedia.org/wiki/Word_error_rate ?
    
    sequences of integers []
    r --- prediction
    h --- ground truth

    TODO: check for correctness 
    TODO: consider adding assertions

    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1)) # why not create coorect dims at start?
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def preds_to_integer(Preds, p_tresh):
    '''`
    Preds --- (log?) probabilities [T, C]  (after batch element selection)
    '''
    preds = torch.argmax(Preds,dim=1).detach().cpu().numpy()
    
    # take maximally likely characters
    probs=np.exp(
        np.max(Preds.detach().cpu().numpy(), axis=1)
        )

    preds=preds.tolist()
    

    out=[]
    for i in range(len(preds)):
        '''
        if char is not eps
        anf if char is not the same as previous one
        '''
        if preds[i] != 0 and preds[i] != preds[i - 1]:
            if probs[i] > p_tresh: # if the character is likely enough
                out.append(preds[i])

    return out 

def wer_eval(preds,labels):
    '''
    word(?) error evaluation
    '''
    preds=preds_to_integer(preds)
    we=wer(preds,labels)

    N = len(preds)
    if N==0:
        return 0

    return we/len(preds)

def show(img):
    npimg = img.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    
def my_collate(batch):
    '''
    After fetching a list of samples using the indices from sampler,
    the function passed as the collate_fn argument is used to collate lists of samples into batches.
    '''
    
    #some shapes 
    one,height,wi,channels=batch[0][0].shape
    #batch size
    batch_size=len(batch)
    #get hte max witdth for padding 
    widths=np.array([x[2] for x in batch])
    max_width=np.max(widths)
    #get label in a long array 
    label_stack=np.concatenate([x[1] for x in batch])
    #get all the lengths 
    length_stack_y=np.array([x[3] for x in batch])
    #Here we will inster images, aka we pad them 
    img_stack=np.zeros(shape=(batch_size,height,max_width,channels))

    #We loop over the batch once 
    for enu,img in enumerate(batch):
        shape=img[2]
        img_stack[enu,:,0:shape,:]=np.squeeze(img[0])
        
    img_stack=torch.tensor(img_stack).float().permute((0,3,1,2))#.cuda()
    label_stack=torch.tensor(label_stack, dtype=torch.int32)#.cuda()
    widths=torch.tensor(widths,dtype=torch.int32)
    length_stack_y=torch.tensor(length_stack_y,dtype=torch.int32)
        
    return img_stack,label_stack,widths,length_stack_y
