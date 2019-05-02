import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_options import TrainOptions
import os
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
import tensorboardX

def main():
    
    opt = TrainOptions()
    args = opt.initialize()
    
    _t = {'iter time' : Timer()}
    
    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)   
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)
    
    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)
    
    model, optimizer = CreateModel(args)
    model_D, optimizer_D = CreateDiscriminator(args)
    
    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])
        
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs", model_name))
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    model_D.train()
    model_D.cuda()
    loss = ['loss_seg_src', 'loss_seg_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real']
    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        
        model.adjust_learning_rate(args, optimizer, i)
        model_D.adjust_learning_rate(args, optimizer_D, i)
        
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        for param in model_D.parameters():
            param.requires_grad = False 
            
        src_img, src_lbl, _, _ = sourceloader_iter.next()
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
        src_seg_score = model(src_img, lbl=src_lbl)
        loss_seg_src = model.loss   
        loss_seg_src.backward()
        
        if args.data_label_folder_target is not None:
            trg_img, trg_lbl, _, _ = targetloader_iter.next()
            trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
            trg_seg_score = model(trg_img, lbl=trg_lbl) 
            loss_seg_trg = model.loss
        else:
            trg_img, _, name = targetloader_iter.next()
            trg_img = Variable(trg_img).cuda()
            trg_seg_score = model(trg_img)
            loss_seg_trg = 0
            
        outD_trg = model_D(F.softmax(trg_seg_score), 0)
        loss_D_trg_fake = model_D.loss
        
        loss_trg = args.lambda_adv_target * loss_D_trg_fake + loss_seg_trg
        loss_trg.backward()
        
        for param in model_D.parameters():
            param.requires_grad = True
        
        src_seg_score, trg_seg_score = src_seg_score.detach(), trg_seg_score.detach()
        
        outD_src = model_D(F.softmax(src_seg_score), 0)
        loss_D_src_real = model_D.loss / 2
        loss_D_src_real.backward()
        
        outD_trg = model_D(F.softmax(trg_seg_score), 1)
        loss_D_trg_real = model_D.loss / 2
        loss_D_trg_real.backward()       
       
        
        optimizer.step()
        optimizer_D.step()
        
        
        for m in loss:
            train_writer.add_scalar(m, eval(m), i+1)
            
        if (i+1) % args.save_pred_every == 0:
            print 'taking snapshot ...'
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source) +str(i+1)+'.pth' ))   
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print '[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff)
            if i + 1 > args.num_steps_stop:
                print 'finish training'
                break
            _t['iter time'].tic()
            
if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))  
    main()
    
    
        