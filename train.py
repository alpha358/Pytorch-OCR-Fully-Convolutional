###########################################################################
############################# General Imports #############################
###########################################################################
from evaluation import wer_eval, preds_to_integer, show, my_collate, AverageMeter
import pdb
import gc  # for ram leak
from IPython import display
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import random


from torch import autograd

import time

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

#Load fake, non handwritten generator
from fake_texts.pytorch_dataset_fake_2 import Dataset
# from IAM_dataset import hwrDataset as Dataset

#Import the loss from baidu
from torch.nn import CTCLoss
# from torch_baidu_ctc import CTCLoss

# from torch.nn.functional import CTCLoss

#Import the model
from fully_conv_model import cnn_attention_ocr

#Helper to count params

@profile
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#Evaluation function preds_to_integer

#Set up Tensorboard writer for current test
writer = SummaryWriter(log_dir="./summary")

# Set up the model
cnn = cnn_attention_ocr(model_dim=64, nclasses=67, n_layers=8)
cnn = cnn.cuda().train()
count_parameters(cnn)

#CTC Loss ,average_frames=True
ctc_loss = CTCLoss(reduction="mean")
#Optimizer: Good initial is 5e5
optimizer = optim.Adam(cnn.parameters(), lr=5e-4)

#We keep track of the Average loss and CER
ave_total_loss = AverageMeter()
CER_total = AverageMeter()

n_iter = 0
batch_size = 6

# torch.save(cnn.state_dict(), "415ksteps_augment_new_gen_e56.pt")

###########################################################################
############################ Dataset Generator ############################
###########################################################################

ds = Dataset()
trainset = DataLoader(dataset=ds,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=0,  # 6 but causes mem leak?
                      pin_memory=False,

                      collate_fn=my_collate)
gen = iter(trainset)

# Learning rate cosine annealing
cs = CosineAnnealingLR(optimizer=optimizer, T_max=250000, eta_min=1e-6)


###########################################################################
################################## Train ##################################
###########################################################################

# leak hunter
# torch.autograd.set_detect_anomaly(True)

@profile
def train_single_epoch(model, gen, optimizer, start_time, curr_best_cer):

    LOSS, ERROR_RATE = [], []
    n_iter = 0
    for i, ge in enumerate(gen):

            if time.time() - start_time > 1*60*60:  # timer
                break

            #to avoid OOM
            if ge[0].shape[3] > 800:
                # width not larger than 800
                continue

            #DONT FORGET THE ZERO GRAD!!!!
            if n_iter % 10:
                optimizer.zero_grad()

            #Get Predictions, permuted for CTC loss
            x = ge[0].cuda()

            log_probs = model(x).permute((2, 0, 1))

            #Targets have to be CPU for baidu loss
            # it is claimed that baidu loss works better
            targets = ge[1].cuda()  # .cpu()

            #Get the Lengths/2 becase this is how much we downsample the width
            input_lengths = ge[2]/2
            target_lengths = ge[3]

            #Get the CTC Loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            #Then backward and step
            loss.backward(retain_graph=False)
            if n_iter % 10:
                optimizer.step()

            LOSS.append(loss.item())

            #Save Loss in averagemeter and write to tensorboard
            ave_total_loss.update(loss.data.item())
            writer.add_scalar("total_loss", ave_total_loss.average(), n_iter)

            # ---- Compute Character error rate
            with torch.no_grad():
                #Here we Calculate the Character error rate
                cum_len = np.cumsum(target_lengths).detach()
                targets = np.split(ge[1].cpu().detach(), cum_len[:-1])
                wer_list = []
                log_probs.detach()
                for j in range(log_probs.shape[1]):
                    wer_list.append(
                        wer_eval(log_probs[:, j, :][0:input_lengths[j], :], targets[j].detach()))

                #if np.average(wer_list)< 10:  # save only good Character error rates ? xD

                CER_total.update(np.average(wer_list))
                writer.add_scalar("CER", CER_total.average(), n_iter)
                ERROR_RATE.append(CER_total.average().item())

                # ----------------- Save Network Parameters ------------------------
                # We save when the new avereage CR is beloew the NPA
                if curr_best_cer > CER_total.average() and CER_total.average() > 0 and CER_total.average() < 1:
                    torch.save(model.state_dict(), "autosave.pt")
                    curr_best_cer = CER_total.average()

                n_iter = n_iter+1
                cs.step()
                lr = optimizer.param_groups[0]["lr"]
                # writer.add_scalar("lr",lr,n_iter)

            # ----------- Clear memory {this was needed} ------------
            del x
            del log_probs
            del input_lengths
            del target_lengths
            del loss
            del targets
            del wer_list
            del ge
            gc.collect()
            # -------------------------------------------------------

            # plot_loss(LOSS, ERROR_RATE)

            if n_iter % 10:
                print('loss: ', LOSS[-1], 'CER: ',ERROR_RATE[-1])

    return np.mean(loss), np.mean(ERROR_RATE)

@profile
def plot_loss(LOSS, ERROR_RATE):
    '''
    Display loss and Error rate
    '''

    # display.clear_output(wait=True)
#     plt.clf()
#     try:
#         fig.clear()
#         plt.close(fig)
#     except:
#         pass

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(LOSS)
    ax[0].set_title('CTC LOSS')
    ax[0].set_xlabel('step')

    ax[1].plot(ERROR_RATE)
    ax[1].set_title('Char Error Rate')
    ax[1].set_xlabel('arb idx')

    fig.show()
    display.display(fig)

    fig.clear()
    plt.close(fig)


npa = 1  # (kas cia ?)


start_time = time.time()

LOSS = []
ERROR_RATE = []  # average character error rate
curr_best_cer = 1

with autograd.detect_anomaly():
    for epochs in range(4):  # 10000

        gen = iter(trainset)
        print("train start")

        loss, error_rate = train_single_epoch(
            cnn, gen, optimizer, start_time, curr_best_cer)
