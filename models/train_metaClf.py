import time
import os

# import scanpy
import numpy as np
import scipy.sparse as sp

import torch
from torch import optim
from torch.utils.data import DataLoader

import models.modelsCNN as modelsCNN
import models.optimizer as optimizer

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import gc
import utils.plot

def train(epoch,trainInput,labels_train,model,optimizer,lossCE,printFreq=500):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    pred = model(trainInput)

    loss=lossCE(pred,labels_train)

    loss.backward()
    optimizer.step()

    if epoch%printFreq==0:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(loss))
    return loss.item()

def plotCTcomp(labels,ctlist,savepath,savenamecluster,byCT,addname='',order=np.array(['Control','AD','FTLD-TDPC','PSP','IPD'])):
    res=np.zeros((order.size,order.size))
    for li in range(res.shape[0]):
        l=order[li]
        nl=np.sum(labels==l)
        ctlist_l=ctlist[labels==l]
        for ci in range(res.shape[1]):
            c=order[ci]
            res[li,ci]=np.sum(ctlist_l==c)
#             res[li,ci]=np.sum(ctlist_l==c)/nl
    if not byCT:
        addname+=''
        for li in range(res.shape[0]):
            l=order[li]
            nl=np.sum(labels==l)
            res[li]=res[li]/nl
    else:
        addname+='_normbyCT'
        for ci in range(res.shape[1]):
            c=order[ci]
            nc=np.sum(ctlist==c)
            res[:,ci]=res[:,ci]/nc
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(res,cmap='binary',vmin=0,vmax=1)
    fig.colorbar(im)
    ax.set_yticks(np.arange(order.size))
    ax.set_yticklabels(order)
    ax.set_xticks(np.arange(order.size))
    ax.set_xticklabels(order)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,savenamecluster+addname+'.pdf'))
    plt.close()

def train_metaClf(inputAll, pathDiag_unique, labels, logsavepath, modelsavepath, plotsavepath, pIDList, allImgNames, sidx_start, weights=None,use_cuda=True, seed=4, testepoch=3500, epochs=6000, saveFreq=100, lr=0.001, weight_decay=0, batchsize=32, model_str='fc3', fc_dim=256,regrs=False,update=False):
    fc_dim1=fc_dim
    fc_dim2=fc_dim
    fc_dim3=fc_dim
    
    if regrs:
        nclasses=1
    else:
        nclasses=np.unique(labels).size
    
    predtest=np.zeros((inputAll.shape[0],nclasses))

    inputAll=torch.tensor(inputAll).cuda().float()
    if regrs:
        labels=torch.tensor(labels.reshape(-1,1)).cuda().float()
    else:
        labels=torch.tensor(labels).cuda().long()

    for patientIDX in range(np.unique(pIDList).size):
        patientID=np.unique(pIDList)[patientIDX]
        if os.path.exists(os.path.join(logsavepath,patientID+'_train_loss')) and (not update):
            continue
        print(patientID,patientIDX)
        sampleIdx=np.arange(inputAll.shape[0])[pIDList==patientID]
        trainIdx=np.arange(inputAll.shape[0])[pIDList!=patientID]

        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)

        nfeatures=inputAll.shape[1]
        if model_str=='fc3':
            model = modelsCNN.FC_l3(nfeatures,fc_dim1,fc_dim2,fc_dim3,nclasses,0.5,regrs=regrs)
            
        if model_str=='fc5':
            model = modelsCNN.FC_l5(nfeatures,fc_dim1,fc_dim2,fc_dim3,fc_dim4,fc_dim5,nclasses,0.5,regrs=regrs)
        if model_str=='fc1':
            model = modelsCNN.FC_l1(nfeatures,fc_dim1,nclasses,regrs=regrs)
        if model_str=='fc0':
            model = modelsCNN.FC_l0(nfeatures,nclasses,regrs=regrs)

        if regrs:
            lossCE=torch.nn.MSELoss()
        else:
            lossCE=torch.nn.CrossEntropyLoss(torch.tensor(weights).cuda().float())
        if use_cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loss_ep=[None]*epochs
        val_loss_ep=[None]*epochs
        t_ep=time.time()

        for ep in range(epochs):
            train_loss_ep[ep]=train(ep,inputAll[trainIdx],labels[trainIdx],model,optimizer,lossCE)


            if ep%saveFreq == 0 and ep!=0:
                torch.save(model.cpu().state_dict(), os.path.join(modelsavepath,patientID+'_'+str(ep)+'.pt'))
            if use_cuda:
                model.cuda()
                torch.cuda.empty_cache()
        print(' total time: {:.4f}s'.format(time.time() - t_ep))

        with open(os.path.join(logsavepath,patientID+'_train_loss'), 'wb') as output:
            pickle.dump(train_loss_ep, output, pickle.HIGHEST_PROTOCOL)

        model.load_state_dict(torch.load(os.path.join(modelsavepath,patientID+'_'+str(testepoch)+'.pt')))
        with torch.no_grad():
            model.cuda()
            model.eval()
            pred = model(inputAll[[sampleIdx]])
            predtest[sampleIdx]=pred.cpu().detach().numpy()

            loss_test=lossCE(pred,labels[[sampleIdx]]).item()

        print(loss_test)
    
    if not os.path.exists(os.path.join(logsavepath,'crossVal_loss')):
        with open(os.path.join(logsavepath,'crossVal_loss'), 'wb') as output:
            pickle.dump(predtest, output, pickle.HIGHEST_PROTOCOL)
        
    nfeatures=inputAll.shape[1]
    if model_str=='fc3':
        model = modelsCNN.FC_l3(nfeatures,fc_dim1,fc_dim2,fc_dim3,nclasses,0.5,regrs=regrs)

    if model_str=='fc5':
        model = modelsCNN.FC_l5(nfeatures,fc_dim1,fc_dim2,fc_dim3,fc_dim4,fc_dim5,nclasses,0.5,regrs=regrs)
    if model_str=='fc1':
        model = modelsCNN.FC_l1(nfeatures,fc_dim1,nclasses,regrs=regrs)
    if model_str=='fc0':
        model = modelsCNN.FC_l0(nfeatures,nclasses,regrs=regrs)
    if regrs:
        lossCE=torch.nn.MSELoss()
    else:
        lossCE=torch.nn.CrossEntropyLoss(torch.tensor(weights).cuda().float())


    for testepoch in range(2000,4000,500):
        if os.path.exists(os.path.join(plotsavepath,'predictions'+str(testepoch)+'.csv')):
            continue
        print(testepoch)
        predtest=np.zeros((inputAll.shape[0],nclasses))

        for patientIDX in range(np.unique(pIDList).size):
            patientID=np.unique(pIDList)[patientIDX]
    #         print(patientID,patientIDX)
            sampleIdx=np.arange(inputAll.shape[0])[pIDList==patientID]
            trainIdx=np.arange(inputAll.shape[0])[pIDList!=patientID]

            seed=3
            torch.manual_seed(seed)
            if use_cuda:
                torch.cuda.manual_seed(seed)

            nfeatures=inputAll.shape[1]

            if use_cuda:
                model.cuda()

            model.load_state_dict(torch.load(os.path.join(modelsavepath,patientID+'_'+str(testepoch)+'.pt')))
            with torch.no_grad():
                model.cuda()
                model.eval()
                pred = model(inputAll[[sampleIdx]])
                predtest[sampleIdx]=pred.cpu().detach().numpy()

                loss_test=lossCE(pred,labels[[sampleIdx]]).item()
        if not regrs:
            predtest_label=np.argmax(predtest,axis=1)
            print(np.sum(predtest_label==labels.cpu().numpy()))
            predtest_label=pathDiag_unique[predtest_label]
            true_label=pathDiag_unique[labels.cpu().numpy()]
        else:
            predtest_label=predtest.flatten()
            true_label=labels.cpu().numpy().flatten()
        res= pd.DataFrame({'sampleName':allImgNames[sidx_start], 'true':true_label, 'predicted':predtest_label})
        res.to_csv(os.path.join(plotsavepath,'predictions'+str(testepoch)+'.csv'))
        if not regrs:
            plotCTcomp(true_label, predtest_label, plotsavepath, 'confusion'+str(testepoch),False)
        else:
            plt.scatter(true_label,predtest_label)
            plt.savefig(os.path.join(plotsavepath,'scatter'+str(testepoch)+'.pdf'))
            plt.close()