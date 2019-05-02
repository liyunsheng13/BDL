import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from options.test_options import TestOptions
from data import CreateTrgDataSSLLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateSSLModel

def main():
    opt = TestOptions()
    args = opt.initialize()    

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = CreateSSLModel(args)

    model.eval()
    model.cuda()   
    targetloader = CreateTrgDataSSLLoader(args)

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []
    
    for index, batch in enumerate(targetloader):
        if index % 100 == 0:
            print '%d processd' % index
        image, _, name = batch
        output = model(Variable(image).cuda(), ssl=True)
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    print thres
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print thres
    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (args.save, name)) 
    
    
if __name__ == '__main__':

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))  
    main()
    