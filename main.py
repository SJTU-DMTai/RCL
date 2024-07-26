import os
import time
import numpy as np
import torch
from info_nce import info_nce
import cupy as cp
import math

import argparse

from model import SASRec
from utils import *
import torch.nn.functional as F

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', type=str)
parser.add_argument('--train_dir', default='wzk', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--neg_size', default=1, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--ssl', default=8, type=int)
parser.add_argument('--start_sim', default=5, type=int)
parser.add_argument('--perc', default=95, type=int)
parser.add_argument('--neg_perc1', default=90, type=int)
parser.add_argument('--neg_perc2', default=80, type=int)
parser.add_argument('--smooth', default=0, type=float)
parser.add_argument('--seed', default=3, type=int)
parser.add_argument('--ngram', default=2, type=int)
parser.add_argument('--max', default=1, type=int)
parser.add_argument('--neg', default=1, type=int)
parser.add_argument('--smooth_loss', default=0.1, type=float)




args = parser.parse_args()
SEED = args.seed
random.seed(SEED) 
os.environ['PYTHONHASHSEED']=f'{SEED}'
np.random.seed(SEED) 
torch.manual_seed(SEED) 
torch.cuda.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    mat_user_train = np.zeros([usernum+1,args.maxlen],dtype=np.int32)
    for u in user_train:
        temp = list(reversed(user_train[u]))
        temp = temp[:args.maxlen] if len(temp)>args.maxlen else temp
        temp = list(reversed(temp))
        mat_user_train[u,args.maxlen-len(temp):]=temp
    
    if args.ssl==11:
        ngrams = []
        ngrams.append([])
        for i in range(1,usernum+1):
            ngram= []
            for j in range(args.maxlen+1-args.ngram):
                gram = []
                for k in range(args.ngram):
                    gram.append(int(mat_user_train[i,j+k]))
                gram = tuple(gram)
                ngram.append(gram)
            ngram = set(ngram)
            ngrams.append(ngram)


    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, f'log_{args.ssl}.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        # try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
        # except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        #     print('failed loading state_dicts, pls check file path: ', end="")
        #     print(args.state_dict_path)
        #     print('pdb enabled for your quick check, pls type exit() if you do not need it')
        #     import pdb; pdb.set_trace()

    cnt = {}     
    with open(f'/root/SSL/SASRec.pytorch/data/{args.dataset}.txt','r') as f2:
        for line in f2:
            it = int(line.strip().split(' ')[1])
            if it in cnt:
                cnt[it] += 1
            else:
                cnt[it] = 1
        freq = np.zeros(itemnum+1)
        freq[0] = 1.0
        for i in cnt:
            freq[i] = cnt[i]
    
    if args.inference_only:
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.decomposition import TruncatedSVD

        embedding_matrix = model.item_emb.weight[1:].cpu().detach().numpy()
        svd = TruncatedSVD(n_components=2)
        svd.fit(embedding_matrix)
        comp_tr = np.transpose(svd.components_)
        proj = np.dot(embedding_matrix, comp_tr)

        # cnt = {}
        # for i in dataset['item_id']:
        #     if i.item() in cnt:
        #         cnt[i.item()] += 1
        #     else:
        #         cnt[i.item()] = 1

        # freq = np.zeros(embedding_matrix.shape[0])
        # for i in cnt:
        #     freq[i-1] = cnt[i]

        # # freq /= freq.max()

        sns.set(style='darkgrid')
        sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
        plt.figure(figsize=(6, 4.5))
        plt.scatter(proj[:, 0], proj[:, 1], s=1)
        plt.colorbar()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        # plt.axis('square')
        # plt.show()
        plt.savefig(f'{args.ssl}.pdf', format='pdf', transparent=False, bbox_inches='tight')



        # from sklearn.decomposition import PCA
        # import pandas as pd
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # item_embed = model.item_emb.weight.cpu().detach().numpy()
        # u,s,vh = np.linalg.svd(item_embed)
        # print(s)
        # plt.plot(s)
        # plt.savefig(f'{args.ssl}.jpg')
        # tsne_result = PCA().fit_transform(item_embed)
        # tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1]})
        # fig, ax = plt.subplots(1)
        # sns.scatterplot(x='tsne_1', y='tsne_2', data=tsne_result_df, ax=ax,s=120)
        # lim = (tsne_result.min()-1, tsne_result.max()+1)
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        # ax.set_aspect('equal')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # plt.savefig(f'{args.state_dict_path}.jpg')

        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

        
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    # U_rep = torch.zeros((usernum+1,args.hidden_units),device=args.device)
    
    U_rep = np.zeros([usernum+1,args.hidden_units])
    # if args.ssl>=8:
    try:
        same_y = np.load(f'same_y_{args.dataset}_95_3_ssl=8.npy')
        global_pos=np.load(f'sim_mat_{args.dataset}_95_3_ssl=8.npy')
    except:
        pass
    
    if args.ssl==6 or args.ssl==8:
        # seq2 = cp.asarray(mat_user_train.reshape(-1))
        # seq_one_hot = cp.zeros((seq2.shape[0], itemnum+1))
        # seq_one_hot[cp.arange(seq2.shape[0]),seq2]=1
        # seq_one_hot = seq_one_hot.reshape(-1,args.maxlen,itemnum+1)
        # seq_sum = cp.sum(seq_one_hot,axis=1)
        # seq_bool = cp.array(seq_sum>0,dtype=int)
        # seq_intersec = cp.matmul(seq_bool, seq_bool.T)  
        # row, col = cp.diag_indices_from(seq_intersec)
        # seq_intersec[row,col] = 0 
        # seq_intersec = seq_intersec.get() 
        if os.path.exists(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy'):
            seq_intersec = np.load(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy')
            same_y = np.load(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy')
            global_pos=np.load(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy')
            global_hard_neg=np.load(f'hard_neg_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy')
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_hard_neg_perc1 = np.percentile(seq_intersec,args.neg_perc1)
            global_hard_neg_perc2 = np.percentile(seq_intersec,args.neg_perc2)
            global_mask = (seq_intersec>global_sim_perc)
            # from collections import Counter
            # print('Counter(data)\n',Counter(global_pos))
        else:
            xxx = 0
            seq_intersec = np.zeros((usernum+1,usernum+1),dtype=int)
            same_y = np.zeros((usernum+1,usernum+1),dtype=int)
            for i in range(1,usernum+1):
                if i%1000==0:
                    print(i)
                for j in range(1,usernum+1):
                    if j==i:
                        continue
                    c = np.intersect1d(mat_user_train[i],mat_user_train[j])
                    seq_intersec[i][j]=len(c)
                    seq_intersec[j][i]=len(c)
                    same_y[i][j]=(mat_user_train[i,-1]==mat_user_train[j,-1])
                    same_y[j][i]=same_y[i][j]
                    if xxx<10:
                        print(len(c))
                        xxx+=1
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            global_pos = []
            for i in range(len(global_mask)):
                if sum(global_mask[i]):
                    global_pos.append(np.random.choice(len(global_mask),1,p=global_mask[i]/sum(global_mask[i]))[0])
                else:
                    global_pos.append(i)
            global_pos = np.array(global_pos,dtype=np.int32)

            
            global_hard_neg_perc1 = np.percentile(seq_intersec,args.neg_perc1)
            global_hard_neg_perc2 = np.percentile(seq_intersec,args.neg_perc2)
            global_mask_neg = (seq_intersec>global_hard_neg_perc2)&(seq_intersec<global_hard_neg_perc1)
            global_hard_neg = []
            for i in range(len(global_mask_neg)):
                if sum(global_mask_neg[i]):
                    temp = []
                    for j in range(args.neg_size):
                        temp.append(np.random.choice(len(global_mask_neg),1,p=global_mask_neg[i]/sum(global_mask_neg[i]))[0])
                else:
                    temp = []
                    for j in range(args.neg_size):
                        temp.append(random.randint(1,usernum))
                global_hard_neg.append(temp)
            global_hard_neg = np.array(global_hard_neg,dtype=np.int32)

            np.save(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy',seq_intersec)
            np.save(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy',same_y)
            np.save(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy',global_pos)
            np.save(f'hard_neg_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_negsize={args.neg_size}.npy',global_hard_neg)

    if args.ssl==9:
        if os.path.exists(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}.npy'):
            seq_intersec = np.load(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}.npy')
            same_y = np.load(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy')
            global_pos=np.load(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}.npy')
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            # from collections import Counter
            # print('Counter(data)\n',Counter(global_pos))
        else:
            xxx=0
            seq_intersec = np.zeros((usernum+1,usernum+1),dtype=int)
            same_y = np.zeros((usernum+1,usernum+1),dtype=int)
            for i in range(1,usernum+1):
                if i%1000==0:
                    print(i)
                for j in range(1,usernum+1):
                    if j==i:
                        continue
                    c = np.intersect1d(mat_user_train[i],mat_user_train[j])
                    # un = np.union1d(mat_user_train[i],mat_user_train[j])
                    weighted_jaccard = sum(np.log(usernum/freq[c])**args.smooth)#/sum(1/freq[un-1])
                    seq_intersec[i][j]=weighted_jaccard
                    seq_intersec[j][i]=weighted_jaccard
                    if xxx==0:
                        print(weighted_jaccard)
                        xxx+=1
                    same_y[i][j]=(mat_user_train[i,-1]==mat_user_train[j,-1])
                    same_y[j][i]=same_y[i][j]
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            global_pos = []
            for i in range(len(global_mask)):
                if sum(global_mask[i]):
                    global_pos.append(np.random.choice(len(global_mask),1,p=global_mask[i]/sum(global_mask[i]))[0])
                else:
                    global_pos.append(i)
            global_pos = np.array(global_pos,dtype=np.int32)
            np.save(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}.npy',seq_intersec)
            np.save(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy',same_y)
            np.save(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}.npy',global_pos)

    if args.ssl==11:
        if os.path.exists(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}_ngram={args.ngram}.npy'):
            seq_intersec = np.load(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}_ngram={args.ngram}.npy')
            same_y = np.load(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy')
            global_pos=np.load(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}_ngram={args.ngram}.npy')
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            # from collections import Counter
            # print('Counter(data)\n',Counter(global_pos))
        else:
            xxx=0
            seq_intersec = np.zeros((usernum+1,usernum+1),dtype=int)
            same_y = np.zeros((usernum+1,usernum+1),dtype=int)
            for i in range(1,usernum+1):
                if i%1000==0:
                    print(i)
                for j in range(1,usernum+1):
                    if j==i:
                        continue
                    c = np.intersect1d(mat_user_train[i],mat_user_train[j])
                    d = ngrams[i]&ngrams[j]
                    val = round((1-args.smooth)*len(d)+args.smooth*len(c))
                    seq_intersec[j][i]= val
                    seq_intersec[i][j]= val
                    if xxx<10:
                        print(len(c),len(d),val)
                        xxx+=1
                    same_y[i][j]=(mat_user_train[i,-1]==mat_user_train[j,-1])
                    same_y[j][i]=same_y[i][j]
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            global_pos = []
            for i in range(len(global_mask)):
                if sum(global_mask[i]):
                    global_pos.append(np.random.choice(len(global_mask),1,p=global_mask[i]/sum(global_mask[i]))[0])
                else:
                    global_pos.append(i)
            global_pos = np.array(global_pos,dtype=np.int32)
            np.save(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}_ngram={args.ngram}.npy',seq_intersec)
            np.save(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy',same_y)
            np.save(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}_smooth={args.smooth}_ngram={args.ngram}.npy',global_pos)
    
    if args.ssl==10:
        if os.path.exists(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy'):
            seq_intersec = np.load(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy')
            same_y = np.load(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy')
            global_pos=np.load(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy')
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            # from collections import Counter
            # print('Counter(data)\n',Counter(global_pos))
        else:
            import Levenshtein
            seq_intersec = np.zeros((usernum+1,usernum+1),dtype=int)
            same_y = np.zeros((usernum+1,usernum+1),dtype=int)
            for i in range(1,usernum+1):
                if i%1000==0:
                    print(i)
                for j in range(1,usernum+1):
                    if j==i:
                        continue
                    x = mat_user_train[i][mat_user_train[i]>0]
                    y = mat_user_train[j][mat_user_train[j]>0]
                    c = Levenshtein.ratio(x,y)
                    seq_intersec[i][j]=c
                    seq_intersec[j][i]=c
                    same_y[i][j]=(mat_user_train[i,-1]==mat_user_train[j,-1])
                    same_y[j][i]=same_y[i][j]
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            global_pos = []
            for i in range(len(global_mask)):
                if sum(global_mask[i]):
                    global_pos.append(np.random.choice(len(global_mask),1,p=global_mask[i]/sum(global_mask[i]))[0])
                else:
                    global_pos.append(i)
            global_pos = np.array(global_pos,dtype=np.int32)
            np.save(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy',seq_intersec)
            np.save(f'same_y_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy',same_y)
            np.save(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_ssl={args.ssl}.npy',global_pos)
    
    if args.ssl==7:
        if os.path.exists(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_{args.smooth}_w.npy'):
            global_pos=np.load(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_{args.smooth}_w.npy')
            from collections import Counter
            print('Counter(data)\n',Counter(global_pos))
        else:
            seq_intersec = np.zeros((usernum+1,usernum+1),dtype=float)
            for i in range(1,usernum+1):
                if i%1000==0:
                    print(i)
                for j in range(1,usernum+1):
                    if j==i:
                        continue
                    c = np.intersect1d(mat_user_train[i],mat_user_train[j])
                    c = c[c>0]
                    # un = np.union1d(mat_user_train[i],mat_user_train[j])
                    weighted_jaccard = sum(math.log(usernum/freq[c-1])**args.smooth)#/sum(1/freq[un-1])
                    seq_intersec[i][j]=weighted_jaccard
                    seq_intersec[j][i]=weighted_jaccard
            
            global_sim_perc = np.percentile(seq_intersec,args.perc)
            global_mask = (seq_intersec>global_sim_perc)
            seq_intersec[~global_mask]=0
            global_pos = []
            for i in range(len(global_mask)):
                if sum(global_mask[i]):
                    global_pos.append(np.random.choice(len(seq_intersec),1,p=seq_intersec[i]/sum(seq_intersec[i]))[0])
                else:
                    global_pos.append(i)
            global_pos = np.array(global_pos,dtype=np.int32)
            np.save(f'raw_sim_mat_{args.dataset}_{args.perc}_{SEED}_{args.smooth}_w.npy',seq_intersec)
            np.save(f'same_y_{args.dataset}_{args.perc}_{SEED}_w.npy',same_y)
            np.save(f'sim_mat_{args.dataset}_{args.perc}_{SEED}_w.npy',global_pos)


    SEED = args.seed
    random.seed(SEED) 
    os.environ['PYTHONHASHSEED']=f'{SEED}'
    np.random.seed(SEED) 
    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED) 
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.ssl>=8:
            global_y = []
            for i in range(len(same_y)):
                if sum(same_y[i]):
                    global_y.append(np.random.choice(len(same_y),1,p=same_y[i]/sum(same_y[i]))[0])
                else:
                    global_y.append(i)
            global_y = np.array(global_y,dtype=np.int32)
        if epoch>args.start_sim and args.ssl==5: 
            U_rep2 = U_rep/np.linalg.norm(U_rep,1)
            global_sim = U_rep2@U_rep2.T
            global_sim_perc = np.percentile(global_sim,90)
            global_mask = (global_sim>global_sim_perc)
            global_pos = []
            for i in range(len(global_mask)):
                if sum(global_mask[i]):
                    global_pos.append(np.random.choice(len(global_mask),1,p=global_mask[i]/sum(global_mask[i]))[0])
                else:
                    global_pos.append(i)
            global_pos = np.array(global_pos,dtype=np.int32)
        



        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits, U_embed = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[:,-1], pos_labels[:,-1])
            # print(loss)
            loss += bce_criterion(neg_logits[:,-1], neg_labels[:,-1])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            if step==num_batch-1:
                print(loss.item())

            # base ssl loss
            if args.ssl==1:
                _, _, aug_embed1 = model(u, seq, pos, neg)
                _, _, aug_embed2 = model(u, seq, pos, neg)

                query = aug_embed1[:,-1,:]
                positive_key = aug_embed2[:,-1,:]


                logits = query @ positive_key.transpose(-2,-1)
                logits = F.normalize(logits,dim=-1)
                labels = torch.arange(len(query), device=query.device)

                ssl_loss = F.cross_entropy(logits / 0.1, labels, reduction='mean')
                if step==num_batch-1:
                    print(ssl_loss.item())
                loss += ssl_loss * 0.1

            if args.ssl==2:
                _, _, aug_embed1 = model(u, seq, pos, neg)
                _, _, aug_embed2 = model(u, seq, pos, neg)

                query = aug_embed1[:,-1,:]
                positive_key = aug_embed2[:,-1,:]


                logits = query @ positive_key.transpose(-2,-1)
                logits = F.normalize(logits,dim=-1)
                labels = torch.arange(len(query), device=query.device)
                ind = np.zeros((args.batch_size,args.batch_size),dtype=int)
                for i in range(args.batch_size):
                    for j in range(args.batch_size):
                        if j==i:
                            continue
                        c = np.intersect1d(seq[i],seq[j])
                        if len(c)>10:
                            ind[i][j]=1
                            ind[j][i]=1
                ind = np.where(ind!=0)
                logits[ind]=float('-inf')
                ssl_loss = F.cross_entropy(logits / 0.1, labels, reduction='mean')
                if step==num_batch-1:
                    print(ssl_loss.item())
                loss += ssl_loss * 0.1
            
            if args.ssl==3:
                _, _, aug_embed1 = model(u, seq, pos, neg)
                _, _, aug_embed2 = model(u, seq, pos, neg)

                query = aug_embed1[:,-1,:]
                positive_key = aug_embed2[:,-1,:]


                logits = query @ positive_key.transpose(-2,-1)
                logits = F.normalize(logits,dim=-1)
                labels = torch.arange(len(query), device=query.device)
                item_num = seq.max()
                seq2 = cp.asarray(seq.reshape(args.batch_size*args.maxlen))
                seq_one_hot = cp.zeros((args.batch_size*args.maxlen, item_num+1))
                seq_one_hot[cp.arange(args.batch_size*args.maxlen),seq2]=1
                seq_one_hot = seq_one_hot.reshape(args.batch_size,args.maxlen,item_num+1)
                seq_sum = cp.sum(seq_one_hot,axis=1)
                seq_bool = cp.array(seq_sum>0,dtype=int)
                seq_intersec = cp.matmul(seq_bool, seq_bool.T)  
                row, col = cp.diag_indices_from(seq_intersec)
                seq_intersec[row,col] = 0 
                seq_intersec = seq_intersec.get() 
                ind = np.where(seq_intersec>10)
                logits[ind]=float('-inf')
                ssl_loss = F.cross_entropy(logits / 0.1, labels, reduction='mean')
                if step==num_batch-1:
                    print(ssl_loss.item())
                loss += ssl_loss * 0.1
            
            #duorec
            if args.ssl==4:
                _, _, aug_embed1 = model(u, seq, pos, neg)
                _, _, aug_embed2 = model(u, seq, pos, neg)

                query = aug_embed1[:,-1,:]
                positive_key = aug_embed2[:,-1,:]


                logits = query @ positive_key.transpose(-2,-1)
                logits = F.normalize(logits,dim=-1)
                labels = torch.arange(len(query), device=query.device)
                pos_last = pos[:,-1].reshape(1,-1)
                is_same = (pos_last.T==pos_last)
                row, col = np.diag_indices_from(is_same)
                is_same[row,col] = False
                ind = np.where(is_same)
                logits[ind]=float('-inf')
                ssl_loss = F.cross_entropy(logits / 0.1, labels, reduction='mean')
                if step==num_batch-1:
                    print(ssl_loss.item())
                loss += ssl_loss * 0.1

                
            # if args.ssl==5:
            #     _, _, aug_embed1 = model(u, seq, pos, neg)
            #     query = aug_embed1[:,-1,:] 
            #     if epoch>args.start_sim:
            #         aug_users = global_pos[u]
            #         sim_aug_seq = mat_user_train[aug_users]
            #         _, _, sim_aug_embed = model(u, sim_aug_seq, pos, neg)
            #     else:
            #         _, _, sim_aug_embed = model(u, seq, pos, neg)
            #     positive_key = sim_aug_embed[:,-1,:]
            #     logits = query @ positive_key.transpose(-2,-1)
            #     logits = F.normalize(logits,dim=-1)
            #     labels = torch.arange(len(query), device=query.device)
            #     ssl_loss = F.cross_entropy(logits / 0.1, labels, reduction='mean')
            #     if step==num_batch-1:
            #         print(ssl_loss.item())
            #     loss += ssl_loss * 0.1 * epoch/args.num_epochs

            if args.ssl==5 or args.ssl==6 or args.ssl==7 or args.ssl==8 or args.ssl==9 or args.ssl==10 or args.ssl==11:
                _, _, aug_embed1 = model(u, seq, pos, neg)
                query = aug_embed1[:,-1,:] 

                neg_users = global_hard_neg[u].reshape(-1) #128*32
                neg_seq = mat_user_train[neg_users] # (128*32)*200
                neg_seq_embed = model.get_embed(neg_seq)
                neg_key = neg_seq_embed.reshape(args.batch_size,args.neg_size,-1)

                if args.max==1:
                    y_users = global_y[u]
                    y_seq = mat_user_train[y_users]
                    _, _, y_embed = model(u, y_seq, pos, neg)
                    
                    y_positive_key = y_embed[:,-1,:]
                    y_logits = query @ y_positive_key.transpose(-2,-1)
                    neg_logits = torch.sum(y_positive_key.unsqueeze(1) * neg_key,dim=2) #bs*negsize
                    # y_logits = torch.cat([y_logits,neg_logits],dim=1)
                    y_logits = F.normalize(y_logits,dim=-1)
                    labels = torch.arange(len(query), device=query.device)

                    
                    y_loss = F.cross_entropy(y_logits / 0.1, labels, reduction='mean')
                    if step==num_batch-1:
                        print('y_loss',y_loss.item())
                    loss += y_loss * 0.1
                
                aug_users = global_pos[u]
                sim_aug_seq = mat_user_train[aug_users]
                _, _, sim_aug_embed = model(u, sim_aug_seq, pos, neg)
                
                positive_key = sim_aug_embed[:,-1,:]
                neg_logits = torch.sum(positive_key.unsqueeze(1) * neg_key,dim=2) #bs*negsize
                logits = query @ positive_key.transpose(-2,-1)
                # logits = torch.cat([logits,neg_logits],dim=1)
                logits = F.normalize(logits,dim=-1)
                if args.max==1:
                    y_min = torch.diag(y_logits).min()
                    y_min_mask = y_min<logits  # true mean diag need change
                    y_min_mask = y_min_mask*torch.eye(logits.shape[0],dtype=torch.bool,device=args.device)
                    # y_min_mask = y_min_mask*torch.cat([torch.eye(logits.shape[0],dtype=torch.bool,device=args.device),torch.zeros((logits.shape[0],logits.shape[1]-logits.shape[0]),dtype=torch.bool,device=args.device)],dim=1)
                    logits = torch.where(y_min_mask,y_min,logits)
                
                labels = torch.arange(len(query), device=query.device)

                weight = torch.tensor(seq_intersec[u,aug_users],device=query.device,dtype=torch.float)
                weight = weight**args.smooth_loss
                sim_loss = F.cross_entropy(logits / 0.1, labels, reduction='mean',weight=weight)
                if step==num_batch-1:
                    print('sim_loss',sim_loss.item())
                loss += sim_loss * 0.1

                


                

            if args.max==0 and (args.ssl==8 or args.ssl==9 or args.ssl==10 or args.ssl==11):
                strong_pos_mask = torch.tensor(global_mask[u]*same_y[u],dtype=torch.bool,device=args.device)
                strong_pos_mask += torch.eye(strong_pos_mask.shape[1],dtype=torch.bool,device=args.device)[u]
                weak_pos_mask = torch.tensor(~global_mask[u]*same_y[u],dtype=torch.bool,device=args.device)
                weak_neg_mask = torch.tensor(global_mask[u]*~same_y[u],dtype=torch.bool,device=args.device)
                strong_neg_mask = torch.tensor(~global_mask[u]*~same_y[u],dtype=torch.bool,device=args.device)[:,u]
                same_y_mask = torch.tensor(same_y[u],dtype=torch.bool,device=args.device)
                
                global_rep = model.get_embed(mat_user_train[:,:-1])
                u_rep = model.get_embed(seq)
                u_rep = F.normalize(u_rep,dim=-1)
                global_rep =F.normalize(global_rep,dim=-1)
                exp_dot = torch.exp(u_rep@global_rep.T/1)  #batchsize*usernum
                strong_pos_dot = exp_dot*strong_pos_mask
                weak_pos_dot = exp_dot*weak_pos_mask
                weak_neg_dot = exp_dot*weak_neg_mask
                same_y_dot = exp_dot*same_y_mask
                strong_neg_dot = exp_dot[:,u]*strong_neg_mask
                #strong_pos = (strong_pos_dot.sum(dim=-1)+1e-8)/(strong_pos_mask.sum(dim=-1)+1e-8)
                #weak_pos = (weak_pos_dot.sum(dim=-1)+1e-8)#/(weak_pos_mask.sum(dim=-1)+1e-8)
                weak_neg = (weak_neg_dot.sum(dim=-1)+1e-8)/(weak_neg_mask.sum(dim=-1)+1e-8)*args.batch_size
                strong_neg = (strong_neg_dot.sum(dim=-1)+1e-8)/(strong_neg_mask.sum(dim=-1)+1e-8)*args.batch_size

                # log_strong = -torch.log(strong_pos_dot/(weak_pos.view(-1,1))+1e-8)*strong_pos_mask #batchsize*usernum
                # sum_strong = (log_strong.sum(dim=-1)+1e-8)/(strong_pos_mask.sum(dim=-1)+1e-8)  #batch_size
                # loss_1 = torch.nanmean(sum_strong)*0.1

                log_weak = -torch.log(same_y_dot/(weak_neg.view(-1,1))+1e-8)*same_y_mask #batchsize*usernum
                sum_weak = (log_weak.sum(dim=-1)+1e-8)/(same_y_mask.sum(dim=-1)+1e-8)  #batch_size
                loss_2 = torch.nanmean(sum_weak)*0.1

                log_neg = -torch.log(weak_neg_dot/(strong_neg.view(-1,1))+1e-8)*weak_neg_mask #batchsize*usernum
                sum_neg = (log_neg.sum(dim=-1)+1e-8)/(weak_neg_mask.sum(dim=-1)+1e-8)  #batch_size
                loss_3 = 0
                # loss_3 = torch.nanmean(sum_neg)*0.1


                # loss_2 = torch.nanmean(-torch.log(weak_pos/(weak_neg)+1e-8))*1
                # loss_3 = torch.nanmean(-torch.log(weak_neg/strong_neg+1e-8))*1
                if step==num_batch-1:
                    print(loss_2.item())
                    # print(loss_3.item())
                loss += (loss_2+loss_3)

            loss.backward()
            adam_optimizer.step()

            model.eval()
            U_rep[u]=U_embed[:,-1,:].cpu().detach().numpy()
            model.train()

            if step==num_batch-1:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 50 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.ssl={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.ssl)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    print("Done")

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    embedding_matrix = model.item_emb.weight[1:].cpu().detach().numpy()
    print()
    svd = TruncatedSVD(n_components=2)
    svd.fit(embedding_matrix)
    comp_tr = np.transpose(svd.components_)
    proj = np.dot(embedding_matrix, comp_tr)
    
    cnt = {}
    with open(f'/root/SSL/SASRec.pytorch/data/{args.dataset}.txt','r') as f:
        for line in f:
            it = int(line.strip().split(' ')[1])
            if it in cnt:
                cnt[it] += 1
            else:
                cnt[it] = 1
        freq = np.zeros(embedding_matrix.shape[0])
        for i in cnt:
            freq[i-1] = cnt[i]
    # freq /= freq.max()
    # sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.scatter(proj[:, 0], proj[:, 1],c=freq, s=1)
    plt.colorbar()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.axis('square')
    # plt.show()
    fname = 'visualization/dataset={}.seed={}.perf={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.ssl={}.pdf'
    fname = fname.format(args.dataset,SEED,args.perc,args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.ssl)
    plt.savefig(fname, format='pdf', transparent=False, bbox_inches='tight')
    # from scipy.linalg import svdvals
    # svs = svdvals(embedding_matrix)
    # svs /= svs.max()
    # np.save(log_dir + '/sv.npy', svs)

    # sns.set(style='darkgrid')
    # sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    # plt.figure(figsize=(6, 4.5))
    # plt.plot(svs)
    # # plt.show()
    # plt.savefig(log_dir + '/svs.pdf', format='pdf', transparent=False, bbox_inches='tight')
