import logging
import torch
from fedml_api.standalone.fedavg.client import Client
from torch.utils.data import TensorDataset,DataLoader
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering
from sklearn.decomposition import PCA
from torchcluster.zoo.spectrum import SpectrumClustering
# from fedml_api.utils.pca import PCA
from sklearn.mixture import GaussianMixture
# from sklearn import preprocessing
# import numpy as np
# from functools import reduce
import random
import wandb
import time
import collections
import copy
import networkx as nx
# import tensorflow as tf
from time import time
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
# from sklearn import datasets
from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
class Group(object):
    def __init__(self, group_id,args,total_label_num):
        self.group_id = group_id
        self.lack_label = [ i for i in range(total_label_num)]
        self.own_label = []
        self.own_label_num = len(self.own_label)
        self.lack_label_num = len(self.lack_label)
        self.group_clients = []
        self.client_total_num = len(self.group_clients)
        self.byzantine_num = 0
        self.normal_client_num = 0
        self.vec_avg_grad = None
        self.iid_group_flag = False
        self.free = True
        self.TS = 0
        self.sup = 0
        self.vec_grad_no_meanvar = None
        self.ge_flat = False
        self.exsiting_byzantine = False
        self.with_label = True
    # def update_init_group(self,j_c):
    #     self.group_clients.append(j_c)
    #     for l in j_c.labels.numpy().tolist():
    #         if l in self.lack_label:
    #             self.lack_label.remove(l)
    #     self.lack_label_num = len(self.lack_label)
    #     self.client_total_num = len(self.group_clients)
    #     if self.lack_label_num <= self.sup:
    #         self.iid_group_flag = True
    def updata_group(self,join_g):
        if self.with_label:
            for j_c in self.group_clients:
                # self.group_clients.append(j_c)
                for l in j_c.labels.numpy().tolist():
                    if l in self.lack_label:
                        self.own_label.append(l)
                        self.lack_label.remove(l)
            self.lack_label_num = len(self.lack_label)
            self.own_label_num = len(self.own_label)
            self.client_total_num = len(self.group_clients)
            if self.lack_label_num <= self.sup:
                self.iid_group_flag = True


class Group_Manager(object):
    def __init__(self, manager_id):
        self.group_num = 0
        self.byt_group_num = 0
        self.normal_group_num = 0
        self.byt_clients_by_cluster_svd = []
        self.sup = 0
        self.device = None
        self.args = None
        self.test_FLAG = False
        self.group_cos_std = 0
        self.group_cos_sum = 0
        self.iid_degree_control = 0
        self.iid_group_num = 0
        self.Manager_id = manager_id
        self.group_list = []
        self.iid_group_list = []
        self.noniid_group_list = []
        self.group_lack_label_num_dict = {}
        self.group_lack_label_num_dict_key_num = 0
        self.group_weight_list = [1. for i in range(self.group_num)]
        self.with_label = True
        self.atk_state =False
        self.normal_group_clients = []
        self.byt_group_clients = []
    def cos_matric(self,pos_clients,neg_clients):
        matric = []
        for ci in neg_clients:
            d_ci_to_all = []
            for cj in pos_clients:
                d_ci_to_all.append(torch.cosine_similarity(ci.vec_grad_no_meanvar,cj.vec_grad_no_meanvar,0))
            matric.append(torch.stack(d_ci_to_all))
        return torch.stack(matric)

    def dis_matric(self,pos_clients,neg_clients):
        matric = []
        for ci in neg_clients:
            d_ci_to_all = []
            for cj in pos_clients:
                d_ci_to_all.append(torch.norm(ci.vec_grad_no_meanvar-cj.vec_grad_no_meanvar,2))
            matric.append(torch.stack(d_ci_to_all))
        return torch.stack(matric)

    def distance_matric(self,clients):
        matric = []
        for ci in clients:
            d_ci_to_all = []
            for cj in clients:
                d_ci_to_all.append(torch.norm(vectorize_state_dict(ci.grad, self.device) - vectorize_state_dict(cj.grad,self.device),2))
            matric.append(torch.stack(d_ci_to_all))
        return torch.stack(matric)
    def client_mkrum(self,clients):
        cos_matric = self.cos_matric(clients,clients)
        cos = []
        good_num = int(0.9*len(clients))
        for i in range(len(clients)):
            sum = torch.sum(torch.topk(cos_matric[i],dim=0,largest=True,sorted=True,k=len(clients)//2)[0]).item() - 1
            cos.append(sum)
        _,final_good = torch.topk(torch.tensor(cos),k=good_num,largest=True,sorted=True)
        normal_state = []
        byt_state = []
        normal_clients = []
        byzantine_count = 0
        byt_num = 0
        for i, c in enumerate(clients):
            if c.byzantine:
                byt_num +=1
            if i in final_good:
                normal_clients.append(c)
                normal_state.append((i,c.client_idx, c.byzantine))
                if c.byzantine:
                    byzantine_count += 1
            else:
                if c.byzantine:
                    byt_state.append((c.client_idx, c.byzantine))
        self.byt_clients_by_cluster_svd.extend(list(set(clients).difference(set(normal_clients))))
        if self.atk_state==False:
            logging.info("client_mkrum_testing:del_byt:{}/{},remaning:{},input,clients_num:{},deleted_state:{}".format(len(byt_state),byt_num,len(normal_clients),len(clients),byt_state))
    def client_svd(self,clients):
        # print("len(clients)",len(clients))
        model_dim = clients[0].model_dim
        if self.args.dataset == "mnist":
            b = 3000
        else:b = 10000
        coff = 0.8
        niters = 1
        copy_niters = niters
        good = []
        while(niters):
            sample_loc = np.random.choice(range(model_dim), b, replace=False).tolist()
            subsample_grad = []
            for i,c in enumerate(clients):
                subsample_grad.append((c.vec_grad_one*c.grad_norm)[sample_loc])
            # del sample_loc
            stack_grad = torch.stack(subsample_grad)
            # del subsample_grad
            sugradmean = torch.mean(stack_grad,0)
            for i in range(len(clients)):
                stack_grad[i] = stack_grad[i] - sugradmean
            # del sugradmean
            _,_,v = torch.svd(stack_grad)
            v_sample = v.t()[0]
            # del v
            score = []
            for i in range(len(clients)):
                score.append(torch.pow(torch.dot(stack_grad[i],v_sample),2))
            # del v_sample
            score,idx = torch.topk(torch.stack(score),k=round(len(clients)*coff),sorted=True,largest=False,dim=0)
            # print("score",score)
            good.append(idx)
            niters -= 1
        # logging.info("client_svd_good_idx" + str(good))
        final_good = torch.where(torch.bincount(torch.cat(good)) == copy_niters)[0]
        # logging.info("client_final_good"+str(final_good))
        normal_state = []
        byt_state = []
        normal_clients = []
        byzantine_count = 0
        byt_num = 0
        for i, c in enumerate(clients):
            if c.byzantine:
                byt_num +=1
            if i in final_good:
                normal_clients.append(c)
                normal_state.append((i,c.client_idx, c.byzantine))
                if c.byzantine:
                    byzantine_count += 1
            else:
                if c.byzantine:
                    byt_state.append((c.client_idx, c.byzantine))
        self.byt_clients_by_cluster_svd.extend(list(set(clients).difference(set(normal_clients))))
        if self.atk_state==False:
            logging.info("client_svd_testing:del_byt:{}/{},remaning:{},input,clients_num:{},deleted_state:{}".format(len(byt_state),byt_num,len(normal_clients),len(clients),byt_state))
        return normal_clients

    def group_svd(self):
        model_dim = self.group_list[0].vec_avg_grad.numel()
        # print(model_dim)
        if self.args.dataset == "mnist":
            b = 3000
        else:b = 10000
        coff = 0.3
        k = 1
        niters = k
        copy_niters = niters
        good = []
        while(niters):
            sample_loc = np.random.choice(range(model_dim), b, replace=False).tolist()
            subsample_grad = []
            for group in self.group_list:
                subsample_grad.append(group.vec_avg_grad[sample_loc])
            # del sample_loc
            stack_grad = torch.stack(subsample_grad)
            # del subsample_grad
            sugradmean = torch.mean(stack_grad, 0)
            for i in range(self.group_num):
                stack_grad[i] = stack_grad[i] - sugradmean
            # del sugradmean
            _, _, v = torch.svd(stack_grad)
            v_sample = v.t()[0]
            # del v
            # print("vsize",v_sample.size())
            score = []
            for i in range(self.group_num):
                score.append(torch.pow(torch.dot(stack_grad[i], v_sample), 2))
            # del v_sample
            score,idx = torch.topk(torch.stack(score), k= int(coff*self.group_num), sorted=True,
                                        largest=False, dim=0)
            # print("score", score)
            print("idx", idx)
            good.append(idx)
            niters -= 1
        final_good = torch.where(torch.bincount(torch.cat(good)) == copy_niters)[0]
        # print("final_good", final_good)
        normal_state = []
        byt_state = []
        normal_clients = []
        byzantine_count = 0
        byt_num = 0
        for i, group in enumerate(self.group_list):
            if group.exsiting_byzantine:byt_num +=1
            if i in final_good:
                group.ge_flat = True
                normal_clients.append(group)
                self.normal_group_clients.extend(group.group_clients)
                normal_state.append((i,group.group_id, group.exsiting_byzantine))
                if group.exsiting_byzantine: byzantine_count += 1
            else:
                self.byt_group_clients.extend(group.group_clients)
                if group.exsiting_byzantine:
                    byt_state.append((group.group_id, group.exsiting_byzantine))
        if self.atk_state==False:
            logging.info("group_svd_test:del_byt:{}/{},remaining:{},input_group_num:{},deleted_state:{}".format(byzantine_count,byt_num,self.group_num-byzantine_count,self.group_num, byt_state))
    def selected_svd(self,pos_clients,neg_clients):
        model_dim = pos_clients[0].vec_grad_one.numel()
        if self.args.dataset == "mnist":
            b = 3000
        else:b = 10000
        coff = 1
        k = 1
        niters = k
        copy_niters = niters
        good = []
        score_ALL =None
        while(niters):
            sample_loc = np.random.choice(range(model_dim), b, replace=False).tolist()
            subsample_grad = []
            neg_clients_grad = []
            for c in pos_clients:
                subsample_grad.append(c.grad_norm*c.vec_grad_one[sample_loc])
            for c in neg_clients:
                subsample_grad.append(c.grad_norm * c.vec_grad_one[sample_loc])
                neg_clients_grad.append(c.grad_norm*c.vec_grad_one[sample_loc])
            # del sample_loc
            stack_grad = torch.stack(subsample_grad)
            neg_clients_grad = torch.stack(neg_clients_grad)
            # del subsample_grad
            sugradmean = torch.mean(stack_grad, 0)
            for i in range(len(neg_clients)):
                neg_clients_grad[i] = neg_clients_grad[i] - sugradmean
            # del sugradmean
            _, ted, v = torch.svd(neg_clients_grad)
            v_sample = v.t()
            # del v
            # print("vsize",v_sample.size())
            score = []
            for i in range(len(neg_clients)):
                ss = [torch.pow(torch.dot(neg_clients_grad[i], v_sample[j]), 2).item() for j in range(ted.numel()//2)]
                score.append(torch.sum(torch.tensor(ss)))
                # score.append(torch.pow(torch.dot(neg_clients_grad[i], v_sample[j]), 2))
            # del v_sample
            score,idx = torch.topk(torch.stack(score), k= int(coff*len(neg_clients)), sorted=True,
                                        largest=False, dim=0)
            score_ALL = score
            # print("idx", idx)
            good.append(idx)
            niters -= 1
        final_good = torch.where(torch.bincount(torch.cat(good)) == copy_niters)[0]
        for i in final_good:
            c = neg_clients[i.item()]
            print("selected_svd:",[c.byzantine,score_ALL[i]])
        normal_state = []
        byt_state = []
        normal_clients = []
        byzantine_count = 0
        byt_num = 0
        # for i, c in enumerate(neg_clients):

        #     if c.byzantine:byt_num +=1
        #     if i in final_good[:]:
        #         normal_clients.append(c)
        #         normal_state.append((score[i],c.client_idx, c.byzantine))
        #         if c.byzantine: byzantine_count += 1
        #     else:
        #         if c.byzantine:
        #             byt_state.append((c.client_idx, c.byzantine))
        # if self.atk_state==False:
        #     logging.info("sellecte_svd_test:del_byt:{}/{},remaining:{},input_group_num:{},deleted_state:{}".format(byzantine_count,byt_num,len(neg_clients)-byzantine_count,len(neg_clients), byt_state))
    def init_Mnager(self,manager_id):
        self.group_num = 0
        self.Manager_id = manager_id
        self.group_list = []
        self.group_lack_label_num_dict = {}
        self.group_lack_label_num_dict_key_num = 0

    def update_manager(self,join_group):
        self.group_list.append(join_group)
        self.group_num += 1
        if self.with_label:
            for g in self.group_list:
                g.sup = self.sup
            # print("self.group_lack_label_num_dict.keys()",self.group_lack_label_num_dict.keys())
            if join_group.lack_label_num not in self.group_lack_label_num_dict.keys():
                self.group_lack_label_num_dict[join_group.lack_label_num] = []
                self.group_lack_label_num_dict_key_num += 1
                self.group_lack_label_num_dict[join_group.lack_label_num].append(join_group)
            else:
                self.group_lack_label_num_dict[join_group.lack_label_num].append(join_group)
    def sort_label_num_dict_by_keys(self):
        dic = dict()
        for key in sorted(self.group_lack_label_num_dict.keys()):#从小到大排序
            # print("1key",key)
            dic[key] = self.group_lack_label_num_dict[key]
        self.group_lack_label_num_dict = copy.deepcopy(dic)
        del dic
    def show_state(self,group_list):
        for group in group_list:
            group_client_state = []
            for c in group.group_clients:
                group_client_state.append((c.client_idx, c.byzantine))
                if c.byzantine:
                    group.exsiting_byzantine = True
                    group.byzantine_num +=1
                group.normal_client_num +=1
            if self.with_label:
                group_state ={"group_id":group.group_id,"iid_state":group.iid_group_flag,"isbyt":group.exsiting_byzantine,"normal_num":group.normal_client_num,"byt_num":group.byzantine_num,"lack_label_num":group.lack_label_num,"lack_label":group.lack_label}
                group_client_state.append((group.group_id,group.exsiting_byzantine,(group.normal_client_num,group.byzantine_num),group.lack_label_num))
            else:
                group_state ={"group_id":group.group_id,"iid_state":group.iid_group_flag,"isbyt":group.exsiting_byzantine,"normal_num":group.normal_client_num,"byt_num":group.byzantine_num}
                group_client_state.append((group.group_id,group.exsiting_byzantine,group.normal_client_num,group.byzantine_num))
            logging.info("****///group_state///******")
            logging.info(group_state)
            logging.info(group_client_state)
            logging.info("****///group_state///******")
    def show_group_state(self):
        for group in self.group_list:
            if group.iid_group_flag:
                self.iid_group_list.append(group)
                self.iid_group_num +=1
            else:
                self.noniid_group_list.append(group)
        self.show_state(self.iid_group_list)
        self.show_state(self.noniid_group_list)
        group_summary = {"iid_group_num":self.iid_group_num,"noniid_group_num":self.group_num-self.iid_group_num,"atk_group_num":self.byt_group_num}
        logging.info(group_summary)
    def set_group_atk_num(self):
        for g in self.group_list:
            if g.exsiting_byzantine:
                self.byt_group_num += 1
            else: self.normal_group_num += 1
    def compute_group_vec_avg_grad(self):
        for group in self.group_list:
            iid_group_vec_local_danwh_grads = []
            for c in group.group_clients:
                iid_group_vec_local_danwh_grads.append(c.vec_grad_one*c.grad_norm)
            group.vec_avg_grad = torch.mean(torch.stack(iid_group_vec_local_danwh_grads), dim=0)

    def compute_group_vec_avg_grad_one(self):
        for group in self.group_list:
            iid_group_vec_local_danwh_grads = []
            for c in group.group_clients:
                iid_group_vec_local_danwh_grads.append(c.vec_grad_no_meanvar)
            group.vec_avg_grad = torch.mean(torch.stack(iid_group_vec_local_danwh_grads), dim=0)
    def random_group(self):
        count = 0
        coff = 0.3
        for i, group in enumerate(self.group_list):
            y=random.random()#0-1之间抽样随机数
            print("y",y)
            while y== 0.5:
                y = random.random()
            if y<0.5 and count<=int(self.group_num*coff):
                count +=1
                group.ge_flat = True
                self.normal_group_clients.extend(group.group_clients)
            if y > 0.5:
                self.byt_group_clients.extend(group.group_clients)
    def group_dismkrum(self,device):#先不上传
        iid_group_danwh_avg_grads = []
        for group in self.group_list:
            iid_group_vec_local_danwh_grads = []
            for c in group.group_clients:
                iid_group_vec_local_danwh_grads.append(c.vec_grad_no_meanvar)
            group.vec_avg_grad = torch.mean(torch.stack(iid_group_vec_local_danwh_grads), dim=0)
            iid_group_danwh_avg_grads.append(group.vec_avg_grad)
        ####MKrum 想法
        coff = 0.3
        cos_simailariy_danwh_avg_grads = []
        for k in range(self.group_num):
            cos_simailariy_danwh_avg_grads.append(
                [torch.norm(iid_group_danwh_avg_grads[k]-iid_group_danwh_avg_grads[j], 2).item() for j in
                 range(self.group_num)])
        # if self.test_FLAG:
        #     all_g_cos = []
        #     for i in range(self.group_num):
        #         if i < self.group_num-1:
        #             all_g_cos.extend(cos_simailariy_danwh_avg_grads[i][i+1:])
        #     self.group_cos_std = torch.std(torch.tensor(all_g_cos)).item()
        #     self.group_cos_sum = torch.sum(torch.tensor(all_g_cos)).item()
        # logging.info("cos_simailariy_danwh_avg_grads:{}".format(torch.tensor(cos_simailariy_danwh_avg_grads)))
        cos_dwh_desc1,_ = torch.topk(torch.tensor(cos_simailariy_danwh_avg_grads).to(device),
                                 k=int(self.group_num*coff) ,dim=1, largest=False, sorted=True)
        cos_dwh_desc, final_good = torch.topk(torch.sum(cos_dwh_desc1,dim=1),
                                 k=int(self.group_num*coff) ,dim=0, largest=False, sorted=True)
        logging.info("cos_dwh_desc{}".format(cos_dwh_desc))
        logging.info("final_good{}".format(final_good))
        ####MKrum 想法
        normal_state = []
        byt_state = []
        normal_clients = []
        byzantine_count = 0
        byt_num = 0
        for i, group in enumerate(self.group_list):
            if group.exsiting_byzantine:byt_num +=1
            if i in final_good:
                group.ge_flat = True
                normal_clients.append(group)
                self.normal_group_clients.extend(group.group_clients)
                normal_state.append((i,group.group_id, group.exsiting_byzantine))
                if group.exsiting_byzantine: byzantine_count += 1
            else:
                self.byt_group_clients.extend(group.group_clients)
                if group.exsiting_byzantine:
                    byt_state.append((group.group_id, group.exsiting_byzantine))
        if self.atk_state==False:
            logging.info("group_mkrum_test:del_byt:{}/{},remaining:{},input_group_num:{},deleted_state:{}".format(byzantine_count,byt_num,self.group_num-byzantine_count,self.group_num, byt_state))
    def group_mkrum(self,device):#先不上传
        iid_group_danwh_avg_grads = []
        for group in self.group_list:
            iid_group_vec_local_danwh_grads = []
            for c in group.group_clients:
                iid_group_vec_local_danwh_grads.append(c.vec_grad_no_meanvar)
            group.vec_avg_grad = torch.mean(torch.stack(iid_group_vec_local_danwh_grads), dim=0)
            iid_group_danwh_avg_grads.append(group.vec_avg_grad)
        ####MKrum 想法
        coff = 0.3
        cos_simailariy_danwh_avg_grads = []
        for k in range(self.group_num):
            cos_simailariy_danwh_avg_grads.append(
                [torch.cosine_similarity(iid_group_danwh_avg_grads[k], iid_group_danwh_avg_grads[j], dim=0).item() for j in
                 range(self.group_num)])
        # if self.test_FLAG:
        #     all_g_cos = []
        #     for i in range(self.group_num):
        #         if i < self.group_num-1:
        #             all_g_cos.extend(cos_simailariy_danwh_avg_grads[i][i+1:])
        #     self.group_cos_std = torch.std(torch.tensor(all_g_cos)).item()
        #     self.group_cos_sum = torch.sum(torch.tensor(all_g_cos)).item()
        # logging.info("cos_simailariy_danwh_avg_grads:{}".format(torch.tensor(cos_simailariy_danwh_avg_grads)))
        cos_dwh_desc1,_ = torch.topk(torch.tensor(cos_simailariy_danwh_avg_grads).to(device),
                                 k=int(self.group_num*coff) ,dim=1, largest=True, sorted=True)
        cos_dwh_desc, final_good = torch.topk(torch.sum(cos_dwh_desc1,dim=1),
                                 k=int(self.group_num*coff) ,dim=0, largest=True, sorted=True)
        logging.info("cos_dwh_desc{}".format(cos_dwh_desc))
        logging.info("final_good{}".format(final_good))
        ####MKrum 想法
        normal_state = []
        byt_state = []
        normal_clients = []
        byzantine_count = 0
        byt_num = 0
        for i, group in enumerate(self.group_list):
            if group.exsiting_byzantine:byt_num +=1
            if i in final_good:
                group.ge_flat = True
                normal_clients.append(group)
                self.normal_group_clients.extend(group.group_clients)
                normal_state.append((i,group.group_id, group.exsiting_byzantine))
                if group.exsiting_byzantine: byzantine_count += 1
            else:
                self.byt_group_clients.extend(group.group_clients)
                if group.exsiting_byzantine:
                    byt_state.append((group.group_id, group.exsiting_byzantine))
        if self.atk_state==False:
            logging.info("group_mkrum_test:del_byt:{}/{},remaining:{},input_group_num:{},deleted_state:{}".format(byzantine_count,byt_num,self.group_num-byzantine_count,self.group_num, byt_state))

    def group_match(self,total_label_num):
        new_group_list = []
        group_id = 0
        # print("self.group_lack_label_num_dict.keys()",self.group_lack_label_num_dict.keys())
        for count, lack_num in enumerate(self.group_lack_label_num_dict.keys()):
            # glist = self.group_lack_label_num_dict[lack_num]
            # print("glist", glist)
            # group_list = Manager.group_lack_label_num_dict[lack_num]
            # print("lack_num", lack_num)
            if self.group_lack_label_num_dict[lack_num] != []:
                for ser_g_id, serach_g in enumerate(self.group_lack_label_num_dict[lack_num]):
                    # print("ser_g_id", ser_g_id)
                    serach_g.free = False
                    target_lack_num = total_label_num - lack_num
                    # print("target_lack_num", target_lack_num)
                    pre_target_lack_num = target_lack_num
                    pre_lack_label_num = serach_g.lack_label_num
                    drt = False
                    while serach_g.lack_label_num > self.sup:  # 达到iid条件或者遍历一遍找不到满足group,则退出，下一个group找
                        serach_g = Search_matching_group(serach_g, self, target_lack_num)  # 找不到或者找到
                        if serach_g.lack_label_num == pre_lack_label_num:  # 找不到,换下一个找
                            # print("serach_g.lack_label_num == pre_lack_label_num")
                            if drt == False:
                                target_lack_num -= 1
                            else:
                                target_lack_num += 1
                            # print("target_lack_num", target_lack_num)
                            if target_lack_num == 0:
                                drt = True
                                target_lack_num = pre_target_lack_num
                            if target_lack_num == total_label_num:
                                # print("target_lack_num == total_label_num")
                                break
                        else:  # 找到,但还没够iid，继续找
                            # print("not_enough,continue")
                            drt = False
                            target_lack_num = total_label_num - serach_g.lack_label_num
                            pre_lack_label_num = serach_g.lack_label_num
                            pre_target_lack_num = target_lack_num
                    # if self.sup <= 2:#保证iid组都是
                    if serach_g.lack_label_num <= self.iid_degree_control:#iid度控制
                        # print("serach_g.iid_group_flag = True")
                        serach_g.iid_group_flag = True
                        # new_group_list.append(serach_g)
                    serach_g.group_id = group_id
                    group_id += 1
                    serach_g.free = True
                    new_group_list.append(serach_g)
                    # del self.group_lack_label_num_dict[lack_num][ser_g_id]
                    # del Manager.group_lack_label_num_dict[lack_num][ser_g_id]
                # print("count", count)
        return new_group_list
    def group_match_by_cluster(self,Manager,cluster_num_dict,cluster_dict):
        max_cluster_num = max(list(cluster_num_dict.values()))
        min_cluster_num = min(list(cluster_num_dict.values()))
        Manager.with_label = False
        for group_id in range(max_cluster_num):
            group = Group(group_id, 1, 1)
            group.with_label = False
            if group_id < min_cluster_num:
                group.iid_group_flag = True
            for cluster in cluster_dict.keys():
                if cluster_num_dict[cluster]:
                    group.group_clients.append(cluster_dict[cluster][group_id])
                    cluster_num_dict[cluster] -= 1
            Manager.update_manager(join_group=group)
        return Manager
def update_clip_state(round_idx,noraml_clients,byz_clients,defend_name):
    state = {}
    atk_num1 = 0
    atk_num2 = 0
    for c in noraml_clients:
        if c.byzantine:
            atk_num1 += 1
    for c in byz_clients:
        if c.byzantine:
            atk_num2 += 1
    logging.info({"sybil_check_normal_atk_num_in_{}".format(defend_name):atk_num1,"delete_atk_num_in_{}".format(defend_name):atk_num2,"round":round_idx})
    wandb.log({"normal_atk_num_in_{}".format(defend_name):atk_num1,"delete_atk_num_in_{}".format(defend_name):atk_num2,"round":round_idx})
def recover_to_dict(list,model_state_dict,device):
    grad = collections.OrderedDict()
    for key, param in model_state_dict.items():
        gp = list[:param.numel()]
        grad[key] = torch.tensor(gp).to(device).view(param.size())
        del list[:param.numel()]
    return grad
def vectorize_weight(state_dict,device):
    weight_list = []
    for (k, v) in state_dict.items():
        if is_weight_param(k):#??????bias
            weight_list.extend(v.view(1,-1).to(device))
    # print(len(weight_list[0]))
    # print(weight_list)
    return torch.cat(weight_list) #torch.cat???????
def vectorize_state_dict(state_dict,device):
    weight_list = []
    for (k, v) in state_dict.items():
        weight_list.extend(v.view(1,-1).to(device))
    return torch.cat(weight_list) #torch.cat???????

def is_weight_param(k):
    return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)
def undo_batch_dataset(dataset):
    new_dataset_x = []
    new_dataset_x_2 = []
    new_dataset_y = []

    for (x,label) in dataset:
        # print("t",x[0])

        # print("{}label{}".format(idx,label))
        new_dataset_x_2.append(x)
        new_dataset_x.extend(x)
        new_dataset_y.append(label)
    # if new_dataset_x !=[]:
    # print("newdata",len(new_dataset_x),len(new_dataset_x_2),len(new_dataset_y))

    dataset_x = torch.stack(new_dataset_x)
    dataset_y = torch.cat(new_dataset_y)
    dataset_x_2 = torch.cat(new_dataset_x_2)
    # print("newdata_tensor", dataset_x.size()[0], dataset_x_2.size()[0], dataset_y.size()[0])
    return dataset_x,dataset_y #合成了一个大batch
def get_zeno_val_data(val_num,data_x,data_y):
    seed = 3
    print("ww",len(data_x),len(data_y))
    random.seed(seed)
    zeno_x = random.sample(data_x.numpy().tolist(), val_num)
    random.seed(seed)
    zeno_y = random.sample(data_y.numpy().tolist(),val_num)
    return [(torch.tensor(zeno_x),torch.tensor(zeno_y))]
def generate_dataset(args,batch_size,data,labels):
    model = ['resnet56','lr','cnn','mobilenet','resnet18_gn','rnn','vgg11','resnet34','cnncf10',"resnet20"]
    if args.model in model:
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
def Zeno_aggregate(w_locals,device):#????
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            # print("w",w_locals[4])
            if i == 0:
                averaged_params[k] = local_model_params[k].to(device) * w
            else:
                averaged_params[k] += local_model_params[k].to(device) * w
    return averaged_params

def grad_dict_aggregate(args,device,clients):
    global_grad = collections.OrderedDict()
    for key,_ in clients[0].grad.items():
        for i, c in enumerate(clients):
            if i == 0:
                local_grad_params = c.grad[key].to(device).mul(c.weight)
            else:
                local_grad_params += c.grad[key].to(device).mul(c.weight)
        global_grad[key] = local_grad_params
    return global_grad

def Direction_Similarity_C1(vec_a,vec_b):
    return torch.sign(torch.dot(vec_a,vec_b))
def length_Similarity_C2(vec_guiding_gradient,vec_client_gradient):
    guiding_gradient_norm = torch.norm(vec_guiding_gradient)
    client_gradient_norm = torch.norm(vec_client_gradient)
    print("client_norm:" + str(client_gradient_norm))
    print("guiding_norm:" + str(guiding_gradient_norm))
    print("client_gradient_norm / guiding_gradient_norm:",client_gradient_norm / guiding_gradient_norm)
    return client_gradient_norm / guiding_gradient_norm
def clip_model_posioned_client(args,clients,atk_state,round_idx):
    e = {"lr":0.00,"cnn":-0.0003,"resnet56":-0.5,"rnn":-10,"resnet18_gn":-5,
         "mobilenet":-0.1,"cnncf10":-0.005,"resnet20":-3,"vgg11":-0.005}
    if args.model == "cnn" and args.dataset == "mnist":
        if args.defend_type == "Zeno":
            if round_idx < 5: e["cnn"] = -0.5
            if round_idx>=5:e["cnn"] = -0.25
        # if args.defend_type == "Zeno"::
        #     if round_idx < 5: e["cnn"] = -5.5
        #     if round_idx>=5:e["cnn"] = -3.5
    if args.model == "resnet20":
        e["resnet20"] = -5
        # if round_idx < 40:  e["resnet20"] = -8
        # if round_idx >= 40: e["resnet20"] = -8
    if args.model == "rnn":
        if round_idx < 40:  e["rnn"] = -4
        if round_idx >= 40: e["rnn"] = -1
    if args.model == "lr":
        # e["lr"] = -0.00
        if round_idx < 5: e["lr"] = -0.003
        if round_idx>=5:e["lr"] = -0.003
    normal_state = []
    # min_loss = min(loss)
    rd_loss = []
    state = []
    record = []
    list_loc_to_c = {}
    for i, c in enumerate(clients):
        # print("c.aid_loss",c.aid_loss)
        list_loc_to_c[i] = c
        rd_loss.append(c.aid_loss)
        state.append(c.byzantine)
        record.append((c.client_idx, c.byzantine, c.aid_loss))
    if atk_state ==False:
        logging.info("panduan_loss{}".format(record))
    rd_loss = torch.tensor(rd_loss)
    nan_loss_num = torch.where(torch.isnan(rd_loss) == True)[0].numel()
    normal_idx_by_delete_negloss = torch.where(rd_loss>e[args.model])[0]
    # print(normal_idx_by_delete_negloss)
    byt_idx_by_delete_negloss = torch.where(rd_loss<=e[args.model])[0]
    for i in normal_idx_by_delete_negloss:
        i = i.item()
        normal_state.append((list_loc_to_c[i].client_idx,state[i]))
    if atk_state == False:
        logging.info("loss_testing:del_byt:{},remainning:{},input_num:{},selected_state:{}".format(len(byt_idx_by_delete_negloss)+nan_loss_num,len(normal_state),len(clients), normal_state))
    return normal_idx_by_delete_negloss,byt_idx_by_delete_negloss
def Search_matching_in_vertical(c_list,lack_num,lack_label,label_num_dic,total_label_num):
        inteset = []
        if lack_num <1 or lack_num >= total_label_num: return lack_num,lack_label
        else:
            if label_num_dic[lack_num] != []:
                for c_i, c in enumerate(label_num_dic[lack_num]):
                    inteset.append(len(list(set(lack_label).intersection(set(c.labels.numpy().tolist())))))
                # print("inteset",inteset)
                for idx, lenth in enumerate(inteset):
                    if lenth == max(inteset):
                        # print("lenth",lenth)
                        # if label_num_dic[lack_num][idx].client_idx
                        # print("aft_lack_label", lack_label)
                        if lenth > 0:
                            all_labels = label_num_dic[lack_num][idx].labels.numpy().tolist()
                            # print("pre_lack_label",lack_label)
                            for l in all_labels:
                                if l in lack_label:
                                    lack_label.remove(l)
                            c_list.append(label_num_dic[lack_num][idx])
                            del label_num_dic[lack_num][idx] #找到和我缺少的标签有交集的用户，拿去配对，删除
                        lack_num = len(lack_label) #这时确少的标签变少l
                        # print("lack_num = len(lack_label)",lack_num )
                        break
            return lack_num,lack_label
def Search_matching_group(sch_g, group_manager, target_lack_num):
    inteset = []
    if target_lack_num in group_manager.group_lack_label_num_dict.keys():
        for g_i, g in enumerate(group_manager.group_lack_label_num_dict[target_lack_num]):
            # print("sch_g.lack_label",sch_g.lack_label,g.own_label)
            inteset.append(len(list(set(sch_g.lack_label).intersection(set(g.lack_label)))))#找选择组与目标组所缺标签的交集，如果交集为空，则表明两者组合后组合组标签满足iid。
        # print("0inteset",inteset)
        for g_i, lenth in enumerate(inteset):
            # if g_i in group_manager.group_lack_label_num_dict[target_lack_num][g_i]:
                if lenth == torch.min(torch.tensor(inteset)).item():
                    t_g = group_manager.group_lack_label_num_dict[target_lack_num][g_i]
                    # print("min(inteset)", min(inteset), len(t_g.lack_label))
                    # print("group_manager.sup,lenth == 0 and sch_g.iid_group_flag ",group_manager.sup,sch_g.iid_group_flag)
                    new_g_max_lc_num = torch.max(torch.tensor([sch_g.lack_label_num,t_g.lack_label_num])).item()
                    new_g_min_lc_num = torch.min(torch.tensor([sch_g.lack_label_num,t_g.lack_label_num])).item()
                    max_num = new_g_max_lc_num - new_g_min_lc_num
                    if max_num - new_g_max_lc_num  + lenth  <= group_manager.sup :
                        if t_g.iid_group_flag == False and t_g.free == True:#找到一个lenth可以降低配对要求。在第一次无法匹配时lenth<sup
                            # if t_g.group_clients not in sch_g.group_clients:
                            sch_g.group_clients.extend(copy.deepcopy(t_g.group_clients))
                            sch_g.updata_group(t_g)
                            del group_manager.group_lack_label_num_dict[target_lack_num][g_i]  # 找到和我缺少的标签有交集的用户，拿去配对，删除
                            del t_g
                            break
    return sch_g
def main_update_drt(avg_grad_list):
    print("avg_grad_list[0]_numel:",avg_grad_list[0].numel())
    _,loc = torch.topk(torch.stack(avg_grad_list),
                             k=avg_grad_list[0].numel()//2, dim=1, largest=True, sorted=True)
    drt_loc = torch.where(torch.bincount(torch.flatten(loc)) > len(avg_grad_list)-5)[0]
    print("drt_loc_num:",drt_loc.numel())
    return drt_loc
# def devide_group(args,w_locals):
def dvided_group(args,clients):
    group_id = 0
    manager_id = 0
    da_na1 = ["mnist", "cifar10"]  # 后面再加上其他数据集
    da_na2 = ["femnist"]
    da_na3 = ["shakespeare"]
    cidx_to_listloc = {}
    client_labels = {}
    num_str_to_num_dic = {}
    if args.dataset in da_na1:
        for c_num in range(10):
            num_str_to_num_dic[c_num] = c_num
        total_label_num = 10
    if args.dataset in da_na2:
        for c_num in range(62):
            num_str_to_num_dic[c_num] = c_num
        total_label_num = 62
    if args.dataset in da_na3:
        for c_num in range(90):
            num_str_to_num_dic[c_num] = c_num
        total_label_num = 90
    for i,c in enumerate(clients):
        cidx_to_listloc[c.client_idx] = i
        client_labels[i] = copy.deepcopy(clients[i].labels)
    Manager = Group_Manager(manager_id=manager_id)
    for i in range(len(clients)):
        group = Group(group_id, args, total_label_num)
        group.group_clients.append(clients[i])
        group.updata_group(group)
        group_id += 1
        Manager.update_manager(join_group=group)
    Manager.sort_label_num_dict_by_keys()
    pre_group_num = 0

    while(Manager.group_num != pre_group_num and Manager.sup <=2):
        # print("i",i)
        pre_group_num = Manager.group_num
        # print("0before pre_group_num = Manager.group_num",pre_group_num ,Manager.group_num)
        new_group_list = Manager.group_match(total_label_num)
        manager_id = 0
        Manager.init_Mnager(manager_id)
        for g_i, g in enumerate(new_group_list):
            # print("new_group{}_lack_label{}".format(g_i, g.lack_label))
            # group = Group(group_id, args, total_label_num)
            # print("g.group_clients", len(g.group_clients))
            # group.group_clients = g.group_clients
            # group_id += 1
            Manager.update_manager(join_group=g)
        Manager.sort_label_num_dict_by_keys()
        # print("Manager.group_lack_label_num_dict.keys()", Manager.group_lack_label_num_dict.keys())
        if Manager.group_num == pre_group_num:  # 不能再合并了，退出。
            Manager.sup += 1
            for g in Manager.group_list:
                g.sup +=1
            # print("Manager.group_num == pre_group_num", Manager.group_num, pre_group_num)
    return Manager
def dvided_random_group(clients,cluster_num):
    cluster_dict = {}  ##3
    cluster_num_dict = {}
    cluster_num = cluster_num
    for cluster in range(cluster_num):
        cluster_dict[cluster] = []
        cluster_num_dict[cluster] = 0
    for c in clients:
        y = np.random.randint(0,cluster_num)
        cluster_dict[y].append(c)
        cluster_num_dict[y] += 1
    return cluster_num_dict,cluster_dict
def devided_cluster_by_dist(args,clients):
    dataset = []
    dict_loc_to_client = {}
    if args.model == "lr":num = 3000
    else:num = 10000
    random_loc = np.random.choice(range(clients[0].vec_grad_no_meanvar.numel()),num, replace=False).tolist()
    for i, c in enumerate(clients):
        dict_loc_to_client[i] = c
        vec = c.vec_grad_no_meanvar/torch.norm(c.vec_grad_no_meanvar,2)
        dataset.append(vec[random_loc])
        class_num = c.class_num
    del random_loc
    if args.defend_module == "B":
        class_num = class_num-class_num+3


    else:
        if args.model == "lr" and args.dataset == "mnist":
            class_num = 33
        if args.model == "resnet20" and args.dataset == "cifar10":
            class_num = 12
        if args.model == "rnn" and args.dataset == "shakespeare":
            class_num = class_num
        if args.model == "cnn" and args.dataset == "femnist":
            class_num = class_num-10
    cluster = SpectrumClustering(class_num)
    resulting_clusters_label, cluster_center = cluster(torch.stack(dataset))
    cluster_dict = {}  ##3
    cluster_num_dict = {}
    for cluster in range(class_num):
        cluster_dict[cluster] = []
        cluster_num_dict[cluster] = 0
    for loc, cluster in enumerate(resulting_clusters_label):
        cluster = cluster.item()
        cluster_dict[cluster].append(dict_loc_to_client[loc])
        cluster_num_dict[cluster] += 1
    # print("cluster_num_dict.values():", cluster_num_dict.values())
    return cluster_num_dict,cluster_dict

def devided_cluster(args,clients):
    dataset = []
    dict_loc_to_client = {}
    if args.model == "lr":num = 3000
    else:num = 10000
    random_loc = np.random.choice(range(clients[0].model_dim),num, replace=False).tolist()
    for i, c in enumerate(clients):
        dict_loc_to_client[i] = c
        dataset.append(c.vec_grad_one[random_loc])
        class_num = c.class_num
    del random_loc
    if args.defend_module == "B":
        class_num = class_num-class_num+3

    else:
        if args.model == "lr" and args.dataset == "mnist":
            class_num = 33
        if args.model == "resnet20" and args.dataset == "cifar10":
            class_num = 12
        if args.model == "rnn" and args.dataset == "shakespeare":
            class_num = class_num
        if args.model == "cnn" and args.dataset == "femnist":
            class_num = class_num-10
    cluster = SpectrumClustering(class_num)
    resulting_clusters_label, cluster_center = cluster(torch.stack(dataset))
    cluster_dict = {}  ##3
    cluster_num_dict = {}
    for cluster in range(class_num):
        cluster_dict[cluster] = []
        cluster_num_dict[cluster] = 0
    for loc, cluster in enumerate(resulting_clusters_label):
        cluster = cluster.item()
        cluster_dict[cluster].append(dict_loc_to_client[loc])
        cluster_num_dict[cluster] += 1
    # print("cluster_num_dict.values():", cluster_num_dict.values())
    return cluster_num_dict,cluster_dict
def compute_avg_simailariy(args,device,iid_group,iid_group_num,current_model,w_locals):
    # if current_model.device:
    vec_g_m = vectorize_state_dict(current_model,device)
    # vec_or_m = vectorize_state_dict(original_model,device)
    cidx_to_listloc = {}
    client_state = []
    iid_group_avg_grads = []
    iid_group_danwh_avg_grads = []
    iid_group_danwh_avg_grads_origianl = []
    # iid_group_danwh_avg_grads_norm = []
    all_clients_grads = []
    all_clients_danwh_grads = []
    all_clients_grads_norm = []
    # var_and_mean = []
    all_clients_danwh_grads_original = []

    iid_group_danwh_avg_grads_pos = []
    iid_group_danwh_avg_grads_neg = []
    mean_var_mean = {}


    for i,(c,w) in enumerate(w_locals):
        client_state.append((i,c.byzantine))
        cidx_to_listloc[c.client_idx] = i
        vec_c_m = vectorize_state_dict(w,device)
        grad = (vec_g_m - vec_c_m)/args.lr
        # orginal_grad = (vec_or_m - vec_c_m)/args.lr
        # all_clients_grads.append(grad)
        grad_one = grad/torch.norm(grad,2)
        all_clients_danwh_grads.append(grad_one)
        # all_clients_danwh_grads_original.append(orginal_grad.div(torch.norm(orginal_grad,2)))
        all_clients_grads_norm.append(torch.norm(grad,2).item())
        # all_clients_grads_norm.append(torch.norm(grad,2).item())
        # all_model_avg.append(grad)
        # all_model_avg_all_client.append(grad_one)
        # all_model_avg.append(vec_c_m/torch.norm(vec_c_m,2))
    # drt_loc = main_update_drt(iid_group_avg_grads)
    # drt_loc = np.random.choice(range(vec_g_m.numel()),5000,replace = False).tolist()
    for k in range(iid_group_num):
        # iid_group_local_grads = []
        # iid_group_vec_local_grads = []
        iid_group_vec_local_danwh_grads = []
        iid_group_vec_local_danwh_grads_original= []
        for c in iid_group[k]:
            # iid_group_vec_local_grads.append(all_clients_grads[cidx_to_listloc[c.client_idx]])
            iid_group_vec_local_danwh_grads.append(all_clients_danwh_grads[cidx_to_listloc[c.client_idx]])
            # iid_group_vec_local_danwh_grads_original.append(all_clients_danwh_grads_original[cidx_to_listloc[c.client_idx]])
            # iid_group_danwh_avg_grads_norm.append(torch.norm())
            # iid_group_vec_local_ws.append(vec_c_m)
            # iid_group_all_model_avg.append(vec_c_m)
            # iid_group_vec_local_ws.append(vec_c_m)

        # avg_grads = torch.mean(torch.stack(iid_group_vec_local_grads),dim=0)
        danwh_avg_grads = torch.mean(torch.stack(iid_group_vec_local_danwh_grads), dim=0)
        # danwh_avg_grads_original = torch.mean(torch.stack(iid_group_vec_local_danwh_grads_original), dim=0)
        # iid_group_avg_grads.append(avg_grads)
        #yuanlai
        iid_group_danwh_avg_grads.append(danwh_avg_grads)
        # iid_group_danwh_avg_grads_origianl.append(danwh_avg_grads_original)
        #relu
        # iid_group_danwh_avg_grads_pos.append(torch.relu(danwh_avg_grads))
        # iid_group_danwh_avg_grads_neg.append(-torch.relu(-danwh_avg_grads))
        #+-1
        # one = torch.ones_like(danwh_avg_grads).to(device)
        # zero = torch.zeros_like(danwh_avg_grads).to(device)
        # pos_loc = torch.where(danwh_avg_grads >= 0,one,-one)
        # pos_loc = torch.where(danwh_avg_grads != 0,pos_loc,zero)
        # iid_group_danwh_avg_grads_neg.append(pos_loc)
        # print("torch.mean(torch.stack(iid_group_vec_local_ws),dim=1)",torch.mean(torch.stack(iid_group_vec_local_ws),dim=0).size())
        # iid_group_avg_vec_models.append(torch.mean(torch.stack(iid_group_vec_local_ws),dim=0))

    cos_simailariy_all_client_grads = []
    cos_simailariy_all_client_dwh_grads = []
    cos_simailariy_avg_grads = []
    cos_simailariy_danwh_avg_grads = []
    cos_simailariy_danwh_avg_grads_original = []
    # distance = []
    # for k in range(args.client_num_per_round):
    #     # cos_simailariy_all_client_grads.append([torch.cosine_similarity(all_clients_grads[k],all_clients_grads[j],dim=0).item() for j in range(args.client_num_per_round)])
    #     cos_simailariy_all_client_dwh_grads.append(
    #         [torch.cosine_similarity(all_clients_danwh_grads[k], all_clients_danwh_grads[j], dim=0).item() for j in
    #          range(args.client_num_per_round)])
    #     distance.append([torch.norm(all_clients_grads[k]-all_clients_grads[j], p=2).item() for j in
    #          range(args.client_num_per_round)])
    # print("cos_simailariy_all_client_grads: ",cos_simailariy_all_client_dwh_grads)
    # print("all_clients_grads_distance: ", distance)
    # distance = []
    for k in range(iid_group_num):
        # cos_simailariy_avg_grads.append([torch.cosine_similarity(iid_group_avg_grads[k],iid_group_avg_grads[j],dim=0).item() for j in range(iid_group_num)])
        cos_simailariy_danwh_avg_grads.append(
            [torch.cosine_similarity(iid_group_danwh_avg_grads[k], iid_group_danwh_avg_grads[j], dim=0).item() for j in
             range(iid_group_num)])
        # cos_simailariy_danwh_avg_grads_original.append(
        #     [torch.cosine_similarity(iid_group_danwh_avg_grads_origianl[k], iid_group_danwh_avg_grads_origianl[j], dim=0).item() for j in
        #      range(iid_group_num)])
        # distance.append([torch.norm(all_clients_grads[k]-all_clients_grads[j], p=2).item() for j in
        #      range(iid_group_num)])
        # distance.append([torch.norm(iid_group_vec_avg_grads[k]-iid_group_vec_avg_grads[j],p=2).item() for j in range(iid_group_num)])

    # print("cos_simailariy_avg_grads: ",cos_simailariy_avg_grads)
    print("cos_simailariy_danwh_avg_grads: ", cos_simailariy_danwh_avg_grads[:2])
    # print("cos_simailariy_danwh_avg_grads_original: ", cos_simailariy_danwh_avg_grads_original)
    # print("distance", distance)
    # cos_simailariy_avg_grads, _ = torch.topk(torch.tensor(cos_simailariy_avg_grads).to(device),
    #                          k=iid_group_num//2, dim=1, largest=True, sorted=True)
    cos_simailariy_danwh_avg_grads, _ = torch.topk(torch.tensor(cos_simailariy_danwh_avg_grads).to(device),
                             k=iid_group_num//2 ,dim=1, largest=True, sorted=True)
    # cos_simailariy_danwh_avg_grads_original, _ = torch.topk(torch.tensor(cos_simailariy_danwh_avg_grads_original).to(device),
    #                          k=iid_group_num-1, dim=1, largest=True, sorted=True)
    # distance, _ = torch.topk(torch.tensor(distance).to(device),
    #                          k=iid_group_num-1, dim=1, largest=False, sorted=True)
    # cos_no_dwh_desc, cos_no_dwh_avg_grad_idx = torch.topk(torch.sum(cos_simailariy_avg_grads, dim=1),
    #                                                  k=iid_group_num, sorted=True,
    #                                                  largest=True)
    cos_dwh_desc, cos_dwh_avg_grad_idx = torch.topk(torch.sum(cos_simailariy_danwh_avg_grads, dim=1),
                                                     k=iid_group_num, sorted=True,
                                                     largest=True)
    # cos_dwh_desc_or, cos_dwh_avg_grad__or_idx = torch.topk(torch.sum(cos_simailariy_danwh_avg_grads_original, dim=1),
    #                                                  k=iid_group_num, sorted=True,
    #                                                  largest=True)
    # dis_dwh_desc, dis_dwh_avg_grad_idx = torch.topk(torch.sum(distance,dim=1),
    #                                                  k=iid_group_num, sorted=True,
    #                                                  largest=False)
    # print("client_state",client_state)
    # print("cos_no_dwh_desc", cos_no_dwh_desc)
    # print("cos_no_dwh_avg_grad_idx",cos_no_dwh_avg_grad_idx)
    print("cos_dwh_desc",cos_dwh_desc)
    print("cos_dwh_avg_grad_idx",cos_dwh_avg_grad_idx)
    # print("cos_dwh_desc_or",cos_dwh_desc_or)
    # print("cos_dwh_avg_grad__or_idx",cos_dwh_avg_grad__or_idx)
    # print("dis_dwh_desc",dis_dwh_desc)
    # print("dis_dwh_avg_grad_idx", dis_dwh_avg_grad_idx)
    iid_group_norm = []
    for id in cos_dwh_avg_grad_idx[:iid_group_num]:
        norm = []
        for c in iid_group[id.item()]:
            norm.append(all_clients_grads_norm[cidx_to_listloc[c.client_idx]])
        iid_group_norm.append(torch.mean(torch.tensor(norm),0).item())
    dis_dwh_avg_grad_idx = None
    mean_var_mean = None
    return all_clients_danwh_grads,cos_dwh_avg_grad_idx,iid_group_danwh_avg_grads,iid_group_norm,mean_var_mean
# def clustering(self,clients):
def fltrust_train(args,device,model_trainer,sample,current_model):
    sample_num = 100
    data_x, data_y = undo_batch_dataset(sample)
    data_list = get_zeno_val_data(sample_num, data_x, data_y)
    sample_generate = generate_dataset(args, args.batch_size, data_list[0][0], data_list[0][1])
    server_updata = model_trainer.get_server_model(args, copy.deepcopy(current_model), sample_generate, device)
    return server_updata
def lr_schedule(epoch):
  lr = 0.05
  if epoch >700:
      lr = 0.0001
  if epoch > 500:
      lr = 0.0002
  elif epoch > 350:
      lr = 0.0005
  elif epoch > 250:
      lr = 0.001
  elif epoch > 150:
      lr = 0.01
  # print('Learning rate: ', lr)
  return lr
class RobustAggregator(object):
    def __init__(self, args):
        self.norm_bound = args.norm_bound  # for norm diff clipping and weak DP defenses
        self.stddev = args.stddev  # for weak DP defenses
        self.clients_cos_std = 0
        self.group_cos_std = 0
        self.group_cos_sum = 0
        self.clients_cos_sum = 0
        self.group_fenzi = 0
        self.atk_state = False
        self.test_FLAG = False
        self.args =args
    def fedavg(self,round_idx, clients, current_model, args, device):
        # global_grad = collections.OrderedDict()
        # for name,param in current_model.items():
        #     paramters = []
        #     for i, c in enumerate(clients):  # 为一个字典
        #         paramters.append(c.grad[name])
        #     median = torch.median(torch.stack(paramters,0),0)[0]
        #     global_grad[name] = median
        # Max_cos = []
        # Max_dis = []
        # cos_data = []
        # dis_data = []
        # client_to_idx_dict = {}
        # for i ,c1 in enumerate(clients):
        #     Distance_among_users = []
        #     cos_among_users = []
        #     dis_among_users = []
        #     for j, c2 in enumerate(clients):
        #         cos_among_users.append(torch.cosine_similarity(c1.vec_grad_no_meanvar , c2.vec_grad_no_meanvar,0).item())
        #         dis_among_users.append(torch.norm(c1.vec_grad_no_meanvar-c2.vec_grad_no_meanvar,2).item())
        #     cos_data.append(cos_among_users)
        #     dis_data.append(dis_among_users)
        #     max_cos,max_idx = torch.topk(torch.tensor(cos_among_users),largest=True,k=3,sorted=True)
        #     max_dis,max_idx = torch.topk(torch.tensor(dis_among_users), largest=True, k=3, sorted=True)
        #     # print(i,max_cos[1],max_idx[1])
        #     Max_cos.append(max_cos[1].item())
        #     Max_dis.append(max_dis[1].item())
        # n = len(clients)
        # mean_cos = (np.sum(np.array(cos_data))-n)/(n*(n-1))
        # mean_dis = 2*np.sum(np.array(dis_data)) / (n * (n - 1))
        # # print(round_idx,Max_cos)
        # # max_c = np.max(np.array(Max_cos))
        # # max_d = np.max(np.array(Max_dis))
        # # wandb.log({"max_cos": max_c, "round": round_idx})
        # wandb.log({"mean_cos": mean_cos, "round": round_idx})
        # # wandb.log({"max_dis": max_d, "round": round_idx})
        # wandb.log({"mean_dis": mean_dis, "round": round_idx})
        # # np.savetxt("./"+str(args.dataset) +".txt",max_c)


        for c in clients:
            c.weight = 1/args.client_num_per_round
            # c.weight = c.weight/weight_sum
        # print(grad_dict_aggregate(args,device,clients))
        fedavg_dict = grad_dict_aggregate(args, device, clients)
        # cos = torch.cosine_similarity(vectorize_weight(fedavg_dict,device),vectorize_weight(global_grad,device),0)
        # dis = torch.norm(vectorize_weight(fedavg_dict, device)-vectorize_weight(global_grad, device), 2)
        # wandb.log({"cos_f_m": cos.item(), "round": round_idx})
        # wandb.log({"dis_f_m": dis.item(), "round": round_idx})
        return fedavg_dict

    def norm_diff_clipping(self, local_state_dict, global_state_dict):
        vec_local_weight = vectorize_weight(local_state_dict,device)
        vec_global_weight = vectorize_weight(global_state_dict,device)
        # print("vec_local_weight",vec_local_weight)
        # clip the norm diff
        vec_diff = vec_local_weight - vec_global_weight
        weight_diff_norm = torch.norm(vec_diff).item()#torch.norm(tensor,p),??tensor?p???,iterm()??tensor??
        clipped_weight_diff = vec_diff / max(1, weight_diff_norm / self.norm_bound)
        clipped_local_state_dict = load_model_weight_diff(local_state_dict,
                                                          clipped_weight_diff,
                                                          global_state_dict)
        return clipped_local_state_dict #??????????????

    def add_noise(self, local_state_dict, device):
        dp_local_state_dict = {}
        for item_index, (k, v) in enumerate(local_state_dict.items()):  # model.state_dict()????????OrderDict???????????????????????????????
            if is_weight_param(k):
                gaussian_noise = torch.randn(v.size(),  # torch.randn?????????
                                             device=device) * self.stddev
                dp_weight = v.cuda() + gaussian_noise
                dp_local_state_dict[k] = dp_weight
            else:
                dp_local_state_dict[k] = v
        return dp_local_state_dict

    def MKrum(self,round_idx,atk_state,clients,args,device):
        metric = {
            "attacker_num":0 ,
            " success rate":0.}
        idx_to_client = {}
        weight_diff = []
        # vec_global_wgt =vectorize_state_dict(w_global,device)
        for idx,c in enumerate(clients):
            weight_diff.append(c.vec_grad_no_meanvar)
            idx_to_client[idx] = c
        Distance_among_users = []
        for i in range(len(clients)):
            Distance_among_users.append([torch.norm(weight_diff[i] - weight_diff[j],2).item() for j in range(len(clients))])
        # print("Distance_among_users:",Distance_among_users)
        distance,_ = torch.topk(torch.tensor(Distance_among_users).to(device),k = args.client_num_per_round - args.attacker_num - 2 + 1,dim=1,largest=False,sorted=True)
        # cos ,_ = torch.topk(torch.tensor(cos).to(device),k = args.client_num_per_round - args.attacker_num - 2 + 1,dim=1,largest=True,sorted=True)
        normal_distance,normal_clients_idx = torch.topk(torch.sum(distance,dim=1),k = args.client_num_per_round - args.attacker_num,sorted=True,largest=False)
        # normal_cos, normal_clients_cos_idx = torch.topk(torch.sum(cos, dim=1),
        #                                                  k=args.client_num_per_round - args.attacker_num, sorted=True,
        #                                                  largest=True)
        normal_clients_state = []
        for idx in normal_clients_idx:
            idx = idx.item()
            if idx_to_client[idx].byzantine:
                metric["attacker_num"] += 1
            normal_clients_state.append((idx,idx_to_client[idx].client_idx,idx_to_client[idx].byzantine))

        # print("normal_clients_state",normal_clients_state)
        # print("normal_clients_cos_idx",normal_clients_cos_idx)
        if atk_state:return args.attacker_num,metric["attacker_num"]
        metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num+0.00001)
        stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"] ,'round':round_idx}
        wandb.log({"success rate": metric["success rate"], "round": round_idx})
        logging.info(stats)
        normal_client_list = []
        for i in normal_clients_idx:
            normal_client_list.append(idx_to_client[i.item()])
        sample_sum = 0
        for c in normal_client_list:
            sample_sum +=  c.get_sample_number()
        for c in normal_client_list:
            c.weight = c.get_sample_number()/sample_sum
        # print("time_total", time.time() - start_total )
        return grad_dict_aggregate(args,device,normal_client_list)
    ####/////梯度估计版本
    def Zeno(self,round_idx,clients,current_model,sample,args,device,model_trainer,atk_state):
        score = []
        metric = {
            "attacker_num":0 ,
            " success rate":0.}

        global_model = copy.deepcopy(current_model)
        # print("old_global_model", global_model)
        sample_num = 8
        data_x,data_y = undo_batch_dataset(sample)
        zeno_val_data = get_zeno_val_data(sample_num,data_x,data_y)
        (x, labels)= zeno_val_data[0]
        print("zeno_labels",labels)
        # print("current_model_type",type(current_model))
        global_loss = model_trainer.compute_loss(current_model,x,labels,device)
        print("global_loss",global_loss)
        client_dict = {}

        for i,c in enumerate(clients):
            for key, param in current_model.items():
                current_model[key] = global_model[key].to(device) - c.grad[key].to(device)
            client_loss = model_trainer.compute_loss(current_model, x, labels, device)
            # print("client_loss",client_loss)
            logging.info("client_" + str(c.client_idx)+"_by_"+str(c.byzantine) +"_norm: " + str(c.grad_norm) + "loss:" + str(global_loss - client_loss))
            score.append(global_loss - client_loss - args.p * c.grad_norm)
            client_dict[i] = (c.client_idx,c.byzantine)
        score_tensor = torch.tensor(score)
        nan_score_clients = torch.where(torch.isnan(score_tensor) == True)[0]
        sorted_score, normal_clients = torch.topk(score_tensor, k=args.client_num_per_round - args.attacker_num,sorted=True)  # -1????,?????????1?????
        # print("normal_clients",normal_clients)
        normal_client_idx = []
        normal_client_panbie = []
        for i in normal_clients:
            i = i.item()
            if i not in nan_score_clients:
                normal_client_idx.append(client_dict[i][0])
                normal_client_panbie.append(client_dict[i])
                if client_dict[i][1] == True:
                    metric["attacker_num"] += 1
        if atk_state:return args.attacker_num,metric["attacker_num"]
        metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num+0.00001)
        stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"] ,'round':round_idx}
        wandb.log({"success rate": metric["success rate"], "round": round_idx})
        logging.info(stats)
        normal_client_list = []
        for c in clients:
            if c.client_idx in normal_client_idx:
                normal_client_list.append(c)
        for c in normal_client_list:#分配权重
            c.weight = 1.0/len(normal_client_list)
        return grad_dict_aggregate(args,device,normal_client_list)
    def DiverseFL(self,round_idx,gradients,current_model,args,device,model_trainer):

        Benign_client = []
        Byzantine_client = []
        metric = {
            "attacker_num": 0,
            " success rate": 0.}
        #??????
        guding_client = Client(None, None, None, None, args, device,
                 model_trainer)
        for i,(c,g_dict) in enumerate(gradients):
            data_x, data_y = undo_batch_dataset(c.local_training_data)
            # print("data_y",len(data_y),data_y)
            # print("guding_client.update_local_dataset?",c.byzantine)
            guding_val_data = get_zeno_val_data(max(1,round(args.s * len(data_y))), data_x, data_y)
            guding_client.update_local_dataset(None,False,guding_val_data, None, max(1,round(args.s * len(data_y))))
            _,guiding_gradient = guding_client.train(current_model)
            C1 = Direction_Similarity_C1(vectorize_weight(guiding_gradient).to(device),vectorize_weight(g_dict).to(device))
            print("start------*C1",C1,c.client_idx,c.byzantine)
            C2 = length_Similarity_C2(vectorize_weight(guiding_gradient).to(device),vectorize_weight(g_dict).to(device))
            if (C1 > 0) & (C2 > args.k1) & (C2 < args.k2):
                Benign_client.append((c,g_dict))
            else:
                Byzantine_client.append((c,g_dict))

        normal_clients_gradients = []
        for (c,g_dict) in Benign_client: #DiverseFL???????????????
            # print("normal_client_" + str(c.client_idx)+"_by_"+str(c.byzantine))

            if c.byzantine == True:
                # print("*normal_client_" + str(c.client_idx) + "_by_" + str(c.byzantine))
                metric["attacker_num"] += 1
            normal_clients_gradients.append((1,g_dict))

        metric["success rate"] = 1 - metric["attacker_num"] / args.attacker_num
        stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"] ,'round':round_idx}
        # wandb.log({"success rate": metric["success rate"], "round": round_idx})
        logging.info(stats)
        # print("normal_clients_gradients:",normal_clients_gradients)
        normal_clients_gradients_mean_dict = aggregate(normal_clients_gradients)
        for (name,g) in normal_clients_gradients_mean_dict.items():
            if is_weight_param(name):
                current_model[name] = current_model[name].to(device) - torch.mul(normal_clients_gradients_mean_dict[name].to(device) + torch.mul(current_model[name].to(device),args.i) ,args.lr)
            else:current_model[name] = g
        print("end------*C1", C1, c.client_idx,c.byzantine)
        return  current_model
    def test(self,round_idx,rd_loc,rd_drt,round_diff,w_global,local_state_dict,args,device,model_trainer,original_model):
        # start_total = time.time()
        w = []
        w_diff = []
        round_diff[round_idx] = []
        idx_to_client = {}

        vec_global_wgt = vectorize_weight(w_global,device)
        # print("model_num",vec_global_wgt.numel())
        for idx ,(c,state_dict) in enumerate(local_state_dict):
            vec_state_dict = vectorize_weight(state_dict,device)
            w.append((vec_state_dict - vec_global_wgt)[rd_loc])#选择观察参数
            round_diff[round_idx].append(vec_state_dict - vec_global_wgt)
            w_diff.append(vec_state_dict)
            idx_to_client[idx] = c.byzantine
        w_t = torch.stack(w,dim=0).t()
        weight = torch.stack(w_diff,dim=0)
        rdiff = torch.stack(round_diff[round_idx], dim=0)
        drt = torch.sum(torch.sign(rdiff),dim=0)
        rd_drt.append(torch.sum(drt).item())
        cm_loc = torch.where(drt==30)[0]
        rdiff_cmloc = rdiff[:,cm_loc]
        print("cm_loc{}{}".format(cm_loc.numel(),cm_loc))
        # print()
        print("drt_sum_by_colum",drt)
        round_diff[round_idx] = w_t
        ###defend schemce
        #single model attacker
        client_drt_sum = torch.sum(torch.sign(rdiff),dim=1)
        clt_srtd_drt = torch.topk(client_drt_sum,k=args.client_num_per_round,sorted=True)
        print("client_drt_sum", clt_srtd_drt)#取最大的值
        benign_set3 = []
        benign_set1 = clt_srtd_drt[1][:args.client_num_per_round-args.attacker_num].tolist()
        # print("benign_set1",benign_set1)
        normal_clients_state = []
        for idx in benign_set1:
            normal_clients_state.append((idx,idx_to_client[idx]))
        print("benign_set1(drt_sum)",normal_clients_state)
        client_step_sum = torch.sum(torch.abs(rdiff),dim=1)
        clt_srtd_step = torch.topk(client_step_sum, k=args.client_num_per_round, sorted=True)
        print("clt_srtd_step", clt_srtd_step)#取最小的值会导致准确率,才不会导致波动太大
        benign_set2 = clt_srtd_step[1][args.attacker_num:].tolist()
        normal_clients_state = []
        for idx in benign_set2:
            normal_clients_state.append((idx,idx_to_client[idx]))
        print("benign_set2(step_sum)",normal_clients_state)
        #conclude attacker
        Distance_among_users = []
        for i in range(args.client_num_per_round):
            Distance_among_users.append([torch.norm(rdiff_cmloc[i] - rdiff_cmloc[j],2).item() for  j in range(args.client_num_per_round)])
        distance,_ = torch.topk(torch.tensor(Distance_among_users).to(device),k = args.client_num_per_round - args.attacker_num - 2 + 1,dim=1,largest=False,sorted=True)
        normal_distance,normal_clients_idx = torch.topk(torch.sum(distance,dim=1),k = args.client_num_per_round - args.attacker_num,sorted=True,largest=False)
        normal_clients_state = []
        benign_set3 = []
        for idx in normal_clients_idx:
            idx = idx.item()
            benign_set3.append(idx)
            normal_clients_state.append((idx,idx_to_client[idx]))
        print("MKrum_benign:",normal_clients_state)
        benign = set(benign_set1).union(set(benign_set2))
        benign = list(benign.union(set(benign_set3)))
        print("final_benign",benign)
        benign_state = []
        for idx in benign:
            benign_state.append((idx,idx_to_client[idx]))
        print("benign_state",benign_state)
        #无权重聚合////////////////
        for (name, param) in w_global.items():  # 为一个字典
            if is_weight_param(name):
                clip_vl = avaraged_w[num:num + torch.flatten(param).numel()]
                w_global[name] = clip_vl.view(param.size())
                num += torch.flatten(param).numel()
            else: w_global[name] = param
        #####//////分析用户的sigmoid

        return w_local_state_list
    def median(self,round_idx,w_global,clients,args,device,model_trainer):
        global_grad = collections.OrderedDict()
        for name,param in w_global.items():
            paramters = []
            for i, c in enumerate(clients):  # 为一个字典
                paramters.append(c.grad[name])
            median = torch.median(torch.stack(paramters,0),0)[0]
            global_grad[name] = median
        for c in clients:
            c.weight = 1/args.client_num_per_round
            # c.weight = c.weight/weight_sum
        # print(grad_dict_aggregate(args,device,clients))
        fedavg = grad_dict_aggregate(args,device,clients)
        cos = torch.cosine_similarity(vectorize_weight(fedavg,device),vectorize_weight(global_grad,device),0)
        wandb.log({"cos_f_m":cos.item(), "round": round_idx})
        return global_grad
    def resample(self,round_idx,clients,args,device):
        s = 2
        rasample_mean_grad = []
        norm = []
        for t in range(len(clients)):
            c = torch.linspace(0, 0, len(clients), dtype=torch.int)
            for i in range(s):
                j = np.random.choice(range(len(clients)),1)[0]
                if c[j] < s :
                    c[j] += 1
                    if c[j] == s : break
            # print("{}_c:{}".format(t,c))
            sample_vec_grad = []
            for idx,count in enumerate(c):
                if count!=0:
                    for _ in range(count):
                        sample_vec_grad.append(clients[idx].vec_grad_one*clients[idx].grad_norm)
            mean_at_t = torch.mean(torch.stack(sample_vec_grad,0),0)## norm_min=< norm <=2*norm_min
            norm_at_t = torch.norm(mean_at_t,2)
            rasample_mean_grad.append(mean_at_t)##norm = 2*norm_min/num
            norm.append(norm_at_t)
        return rasample_mean_grad,norm
        # ####////////MKrum
        # weight_diff = []
        # vec_global_wgt = vectorize_weight(current_model).to(device)
        # for idx ,(c,state_dict) in enumerate(rasample_model):
        #     vec_state_dict = vectorize_weight(state_dict)
        #     weight_diff.append((vec_state_dict - vec_global_wgt).to(device))
        # Distance_among_users = []
        # for i in range(args.client_num_per_round):
        #     Distance_among_users.append([torch.norm(weight_diff[i] - weight_diff[j],2).item() for  j in range(args.client_num_per_round)])
        # distance,_ = torch.topk(torch.tensor(Distance_among_users).to(device),k = args.client_num_per_round - args.attacker_num - 2 + 1,dim=1,largest=False,sorted=True)
        # normal_distance,normal_clients_idx = torch.topk(torch.sum(distance,dim=1),k = args.client_num_per_round - args.attacker_num,sorted=True,largest=False)
        # normal_client_models = []
        # for i in range(args.client_num_per_round):
        #     if i in normal_clients_idx:
        #         normal_client_models.append(rasample_model[i])
        # return normal_client_models
    def faba(self,round_idx,clients,args,device):
        metric = {
            "attacker_num": 0,
            " success rate": 0.}

        print("len(clients)",len(clients))
        local_updates = np.array(
            torch.stack([c.vec_grad_no_meanvar for c in clients], 0).cpu().tolist())
        print("len(local_updates)", len(local_updates))
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(local_updates)
        y = []
        for c in clients:
            if c.byzantine:
                y.append(0)
            else:y.append(1)
        if round_idx %10 == 0:
            y = np.array(y)
            filename1 = "./hetero1_round_x_reduced.npy"+str(round_idx)
            filename2 = "./hetero1_byt.npy"+str(round_idx)
            # 写文件
            np.save(filename1, X_reduced)
            np.save(filename2, y)

        for k in range(args.client_num_per_round):
            if k < args.attacker_num:
                for c in clients:
                    c.weight = 1 / len(clients)
                mean = grad_dict_aggregate(args,device,clients)
                D_to_mean = []
                vec_mean = vectorize_weight(mean,device)
                for i in range(args.client_num_per_round - k):
                    vec_c_w = clients[i].vec_grad_no_meanvar
                    D_to_mean.append(torch.norm(vec_c_w - vec_mean, 2).item())
                distance, index = torch.topk(torch.tensor(D_to_mean).to(device),
                                         k=1, dim=0, largest=True,
                                         sorted=True)
                # print(distance)
                # print("index",index[0])
                clients.pop(index[0].item())
            else: break

            # print("X_reduced.T[0]",len(X_reduced.T[0]))
        # plt.figure()
        # fig = plt.figure(figsize=(8, 4))
        # ax = fig.add_subplot(122)
        # ax.scatter(X_reduced.T[0], X_reduced.T[1], c=y)
        # plt.show()
        # plt.plot((X_reduced.T[self.args.attacker_num:][0],X_reduced.T[self.args.attacker_num:][1], color="#CD5C5C", linestyle="-", linewidth=3.0, label="sample 2")

        # X = local_updates
        # pca.fit(X)
        # recon = pca.inverse_transform(X_reduced)
        # rmse = [mean_squared_error(local_updates[i], recon[i], squared=False) for i in range(len(local_updates))]
        # print("X_reduced:",X_reduced)

        # estimator = KMeans(n_clusters=2)  # 构造聚类器
        # estimator.fit(X_reduced)  # 聚类
        # center = estimator.cluster_centers_
        # print("center[0]", center[0])
        # print("center[1]", center[1])
        # center_dist = np.linalg.norm(center[0] - center[1])
        # print("center_dist:", center_dist)
        # label_pred = estimator.labels_  # 获取聚类标签
        # selectedId_c1 = []
        # selectedId_c2 = []
        # for i in range(len(selectedId)):
        #     if label_pred[i] == 0:
        #         selectedId_c1.append(selectedId[i])
        #     else:
        #         selectedId_c2.append(selectedId[i])
        # print("selectedId_c1", selectedId_c1)
        # print("selectedId_c2", selectedId_c2)
        # print("label_pred", label_pred)
        # print(len(w_locals))
        for c in clients:
            c.weight = 1/len(clients)
            if c.byzantine:
                metric["attacker_num"] += 1
        metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.000001)
        stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
        logging.info(stats)
        wandb.log({"success rate": metric["success rate"], "round": round_idx})
        return grad_dict_aggregate(args,device,clients)
    def expsmoo(self,round_idx,clients,alpha,s1,s2,args,device,model_trainer):
        metric = {
            "attacker_num": 0,
            " success rate": 0.}
        # alpha = 0.8
        zipped_dis = []
        normal_index = []
        normal_clients = []
        colect_s1_weight = []
        colect_s2_weight = []
        vec_s1_w = vectorize_weight(s1,device)
        vec_s2_w = vectorize_weight(s2,device)
        predict_w = 2*vec_s1_w - vec_s2_w + (alpha/(1-alpha)) * (vec_s1_w - vec_s2_w)
        for idx,c in enumerate(clients):
            d = torch.norm(c.vec_grad_no_meanvar - predict_w, p=2).item()
            zipped_dis.append((idx,d))
        print("distance local model between predicted model:", zipped_dis)
        zipped_dis.sort(key=lambda x: x[1])
        print("distance sorted:", zipped_dis)
        # 找断层位置
        km = KMeans(n_clusters=2)
        km.fit([[zipped_dis[i][1]] for i in range(args.client_num_per_round)])
        fault_value = np.mean(km.cluster_centers_)
        print("fault_value:", fault_value)
        for i in range(args.client_num_per_round):
            if zipped_dis[i][1] < fault_value:
                normal_index.append(zipped_dis[i][0])
            else:
                break
        normal_index.sort()
        print("normal_index:", normal_index, (len(normal_index)))
        for i in normal_index:
            normal_clients.append(clients[i])
            if clients[i].byzantine:
                metric["attacker_num"] += 1
        metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.0001)
        stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
        wandb.log({"success rate": metric["success rate"], "round": round_idx})
        logging.info(stats)
        for c in normal_clients:
            c.weight = 1 / len(normal_clients)
        return grad_dict_aggregate(args, device, normal_clients)
        # return Zeno_aggregate(normal_w, device)
    def fltrust(self,round_idx,clients,current_model,sample,args,device,model_trainer):
        server_updata = fltrust_train(args,device,model_trainer,sample,current_model)
        # vec_s_g = vectorize_state_dict(server_updata,device)
        s_g_norm = torch.norm(vectorize_state_dict(server_updata,device),2)
        vec_s_g_no_meanvar = vectorize_weight(server_updata,device)
        TS = []
        sum_ts = 0
        cos = []
        state = []
        noram_score_mean = 0.
        byt_score_mean = 0.0
        for i,c in enumerate(clients):
            cos.append(torch.cosine_similarity(vec_s_g_no_meanvar,c.vec_grad_no_meanvar,dim=0))
            TS.append(torch.relu(torch.cosine_similarity(vec_s_g_no_meanvar,c.vec_grad_no_meanvar,dim=0)).item())
            if c.byzantine:
                byt_score_mean += TS[i]
            else:noram_score_mean += TS[i]
            state.append((c.byzantine,TS[i]))
            c.weight = (1/c.grad_norm)*s_g_norm*TS[i]
            if i == 0:
                sum_ts = TS[i]
            else:
                sum_ts += TS[i]
        noram_score_mean = noram_score_mean/(args.client_num_per_round - args.attacker_num)
        if args.attacker_num != 0:
            byt_score_mean = byt_score_mean/args.attacker_num
        wandb.log({"noram_score_mean":noram_score_mean,"byt_score_mean": byt_score_mean,"round": round_idx})
        print("cos:",cos)
        # print("dis:",dis)
        # print("TS:",TS)
        print("score",state)
        # print('s_g_norm',s_g_norm)
        for c in clients:
            c.weight = c.weight/sum_ts
        return grad_dict_aggregate(args,device,clients)
    def dnc(self,round_idx, clients, atk_state, args, device):
        metric = {
            "attacker_num": 0,
            " success rate": 0.}
        model_dim = clients[0].model_dim
        if args.dataset == "mnist":
            b = 3000
        else:b = 10000
        coff = 1
        niters = 1
        copy_niters = niters
        good = []
        while(niters):
            sample_loc = np.random.choice(range(model_dim), b, replace=False).tolist()
            subsample_grad = []
            if args.defend_type == "fedbt":
                for grad_vec in clients:
                    subsample_grad.append(grad_vec[sample_loc])
            else:
                for i,c in enumerate(clients):
                    subsample_grad.append(vectorize_state_dict(c.grad,device)[sample_loc])
            stack_grad = torch.stack(subsample_grad)
            sugradmean = torch.mean(stack_grad,0)
            for i in range(len(clients)):
                stack_grad[i] = stack_grad[i] - sugradmean
            # print("stack_sample_size",stack_grad.size())
            u,s,v = torch.svd(stack_grad)
            v_sample = v.t()[0]
            # print("vsize",v_sample.size())
            score = []
            for i in range(len(clients)):
                # print(subsample_grad[i])
                score.append(torch.pow(torch.dot(stack_grad[i],v_sample),2))
            if args.defend_type == "fedbt":
                score, idx = torch.topk(torch.stack(score), k=len(clients)//2, sorted=True,
                                        largest=False, dim=0)
            else:
                score,idx = torch.topk(torch.stack(score),k=len(clients) - args.attacker_num * coff,sorted=True,largest=False,dim=0)
            print("score",score)
            print("idx",idx)
            good.append(idx)
            niters -= 1
        final_good = torch.where(torch.bincount(torch.cat(good)) == copy_niters)[0]
        print("final_good",final_good)
        normal_state = []
        normal_clients = []
        for i, c in enumerate(clients):
            if i in final_good:
                normal_clients.append(c)
                normal_state.append((i,c.client_idx,c.byzantine))
                if c.byzantine:metric["attacker_num"] +=1
        print("normal_state",normal_state)
        if atk_state :return args.attacker_num,metric["attacker_num"]
        else:
            metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
            stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
            logging.info(stats)
            wandb.log({"success rate": metric["success rate"], "round": round_idx})
            for c in normal_clients:
                c.weight = 1/len(normal_clients)
            return grad_dict_aggregate(args,device,normal_clients)
    def fedbt(self,round_idx,clients,atk_state,cur_model,args,device,model_trainer):
        if self.args.defend_module == "ABCD":
        # module ABCD
        #     beta_1 =  ##
        #     cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # print("cos_matix",cos_matric1)


            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = args.attacker_num
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i],dim=0,largest=True,sorted=True,k=sybil_num)[0]).item() - 1
                cos.append(sum)

            cos_sum_mean = torch.tensor(cos)/sybil_num
            descr_cos,descr_cos_idx = torch.topk(cos_sum_mean,k=sybil_num,sorted=True,largest=True)
            # print("cos_sum_mean",cos_sum_mean)
            for i,c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:good_clients_by_clip_sibyl_attacker.append(c)

            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state==False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),len(good_clients_by_clip_sibyl_attacker),args.client_num_per_round))
            #clip_sibyl_attack_by_cos_similarity_end



            cluster_num_dict,cluster_dict = devided_cluster(args,good_clients_by_clip_sibyl_attacker)
            if self.atk_state==False:
                logging.info("cluster_num_dict:"+str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
                    # Manager.client_svd(cluster_dict[key])
                    # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager,cluster_num_dict,cluster_dict)
            if self.atk_state==False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            group_manager.group_mkrum(device)

            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            num= 1
            # num = int(len(group_manager.byt_group_clients) * 0.01)##若设置为0，则计算norm时出错？？
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx],k=len(clients)//4,largest=True,sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))
            Max_Sum = torch.stack(Max_Sum)
            # if num == 0:
            # incres_cos, incres_cos_idx = [],[]
            # else:
            incres_cos,incres_cos_idx =torch.topk(Max_Sum,k=num,largest=False,sorted=True)
            byt_in_group = []
            print("incres_cos_idx",incres_cos_idx)
            for i,idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                byt_in_group.append(copy.deepcopy(group_manager.byt_group_clients[idx]))
                logging.info("byt_in_group:{}".format([c.client_idx,c.byzantine,incres_cos[i]]))
            # norm = []
            # for c in clients:
            #     norm.append(c.grad_norm.item())
            # print("norm", norm)
            # group_normal_clients = list(set(clients))
            ##module: A+B+C+D,beta_1 = 0.20

            # group_normal_clients = []
            # for c in clients:
            #     if c not in byt_in_group and sibyl_atk:
            #         # clients.remove(c)
            #         group_normal_clients.append(c)
            if len(byt_in_group) == 0 and len(sibyl_atk) == 0:group_normal_clients = clients
            else:
                group_normal_clients = list(set(clients).difference(set(byt_in_group)).difference(set(sibyl_atk)))
            # print(group_normal_clients)
            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            # norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx,c.byzantine])
                # norms1.append(c.grad_norm)
            if self.atk_state==False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(len(byt_in_group),args.attacker_num,len(group_normal_clients),
                                                                                len(clients), group_state))
            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss,byt_idx_by_delete_negloss = clip_model_posioned_client(args,group_normal_clients,self.atk_state,round_idx) #h获取没有被模型中毒攻击过的用户
            print("normal_idx_by_delete_negloss",normal_idx_by_delete_negloss)
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(group_normal_clients[idx])
                # print((group_normal_clients[idx].grad_norm))
                norms1.append(group_normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(group_normal_clients[idx])
            if self.atk_state==False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss,"Model_Check")

            "###model_poisoning_defend 2:model_check_end"
            "##model_poisoning_defend 3: model aggregation by same norm begin"

            """"""
            ##clip_large and small norm begin
            ##clip_large and small norm end
            ##clip_large and small norm begin
            print(norms1)
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            norm_clients = []
            norm_byz = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((normal_clients_clip_by_loss[i].byzantine,normal_clients_clip_by_loss[i].client_idx))
                    byzantine_idx.append(normal_clients_clip_by_loss[i].client_idx)
                    norm_byz.append(normal_clients_clip_by_loss[i])
                else:norm_clients.append(normal_clients_clip_by_loss[i])
            if self.atk_state==False:
                update_clip_state(round_idx, norm_clients, norm_byz,"Norm_Check")
            normal_clients_clip_by_norm = []
            for c in normal_clients_clip_by_loss:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(normal_clients_clip_by_loss),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))
            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)
            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = 1.0
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]/len(normal_clients_clip_by_norm)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "BCD":
            #module BCD
            metric = {
                "attacker_num": 0,
                " success rate": 0.}

            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients, clients)
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
            cluster_num_dict, cluster_dict = devided_cluster(args, clients)
            if self.atk_state == False:
                logging.info("cluster_num_dict:" + str(cluster_num_dict.values()))
            group_manager = Manager.group_match_by_cluster(Manager, cluster_num_dict, cluster_dict)
            if self.atk_state == False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            group_manager.group_mkrum(device)
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx], k=len(clients) // 4, largest=True, sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos, incres_cos_idx = torch.topk(Max_Sum, k=args.attacker_num , largest=False, sorted=True)
            byt_in_group = []
            for i, idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx, c.byzantine, incres_cos[i]]))
            # group_normal_clients = list(set(clients))
            ##module: B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)))

            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx, c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state == False:
                logging.info(
                    "group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(len(byt_in_group),
                                                                                                              args.attacker_num,
                                                                                                              len(
                                                                                                                  group_normal_clients),
                                                                                                              len(clients),
                                                                                                              group_state))
            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss, byt_idx_by_delete_negloss = clip_model_posioned_client(args, group_normal_clients,
                                                                                                 self.atk_state,
                                                                                                 round_idx)  # h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(group_normal_clients[idx])
                norms1.append(group_normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(group_normal_clients[idx])
            if self.atk_state == False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss, "Model_Check")

            "###model_poisoning_defend 2:model_check_end"
            "##model_poisoning_defend 3: model aggregation by same norm begin"

            """"""

            ##clip_large and small norm begin
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i, nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median)) > 8 or torch.abs(nm.div(norms_median)) < 0.05:
                    byzantine.append((normal_clients_clip_by_loss[i].byzantine, normal_clients_clip_by_loss[i].client_idx))
                    byzantine_idx.append(normal_clients_clip_by_loss[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in normal_clients_clip_by_loss:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1 / c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info(
                    "norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(
                        len(byzantine), selected_byt_num, len(normal_clients_clip_by_norm), len(normal_clients_clip_by_loss),
                        byzantine))
                wandb.log({"aggregation_total_num": len(normal_clients_clip_by_norm), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))
            for c in normal_clients_clip_by_norm:
                    c.weight = c.weight.mul(norms_mean)
            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean / norms_min  ###不能在聚合后补模，会打乱内部梯度的方向
            # global_norm = 1.0
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key] / len(normal_clients_clip_by_norm)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "ABC":
        #module ABC
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i],dim=0,largest=True,sorted=True,k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos)/sybil_num
            descr_cos,descr_cos_idx = torch.topk(cos_sum_mean,k=sybil_num,sorted=True,largest=True)
            print("cos_sum_mean",cos_sum_mean)
            for i,c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state==False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),len(good_clients_by_clip_sibyl_attacker),args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end



            cluster_num_dict,cluster_dict = devided_cluster(args,good_clients_by_clip_sibyl_attacker)
            if self.atk_state==False:
                logging.info("cluster_num_dict:"+str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
                    # Manager.client_svd(cluster_dict[key])
                    # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager,cluster_num_dict,cluster_dict)
            if self.atk_state==False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            group_manager.group_mkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx],k=len(clients)//4,largest=True,sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos,incres_cos_idx =torch.topk(Max_Sum,k=int(len(group_manager.byt_group_clients) * 0.20
                                                                ),largest=False,sorted=True)
            byt_in_group = []
            for i,idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx,c.byzantine,incres_cos[i]]))
            # group_normal_clients = list(set(clients))
            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)).difference(set(sibyl_atk)))

            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx,c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state==False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(len(byt_in_group),args.attacker_num,len(group_normal_clients),
                                                                                len(clients), group_state))
            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss,byt_idx_by_delete_negloss = clip_model_posioned_client(args,group_normal_clients,self.atk_state,round_idx) #h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(group_normal_clients[idx])
                norms1.append(group_normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(group_normal_clients[idx])
            if self.atk_state==False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss,"Model_Check")

            normal_state = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_loss:
                c.weight = 1/len(normal_clients_clip_by_loss)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_loss),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_loss)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "ACD":
            # module ACD
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i],dim=0,largest=True,sorted=True,k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos)/sybil_num
            descr_cos,descr_cos_idx = torch.topk(cos_sum_mean,k=sybil_num,sorted=True,largest=True)
            print("cos_sum_mean",cos_sum_mean)
            for i,c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state==False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),len(good_clients_by_clip_sibyl_attacker),args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end


            normal_clients = list(set(clients).difference(set(sibyl_atk)))

            #
            # group_state = []
            # """"""
            # norms1 = []
            # """"""
            # for c in group_normal_clients:
            #     group_state.append([c.client_idx,c.byzantine])
            #     norms1.append(c.grad_norm)

            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss,byt_idx_by_delete_negloss = clip_model_posioned_client(args,normal_clients,self.atk_state,round_idx) #h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(normal_clients[idx])
                norms1.append(normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(normal_clients[idx])
            if self.atk_state==False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss,"Model_Check")

            "###model_poisoning_defend 2:model_check_end"
            "##model_poisoning_defend 3: model aggregation by same norm begin"

            """"""
            ##clip_large and small norm begin
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((normal_clients_clip_by_loss[i].byzantine,normal_clients_clip_by_loss[i].client_idx))
                    byzantine_idx.append(normal_clients_clip_by_loss[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in normal_clients_clip_by_loss:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(normal_clients_clip_by_loss),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))

            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean/norms_min###不能在聚合后补模，会打乱内部梯度的方向
            # global_norm = 1.0
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]/len(normal_clients_clip_by_norm)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "ABD":
            # module ABD
            beta_1 = 0.12 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i],dim=0,largest=True,sorted=True,k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos)/sybil_num
            descr_cos,descr_cos_idx = torch.topk(cos_sum_mean,k=sybil_num,sorted=True,largest=True)
            print("cos_sum_mean",cos_sum_mean)
            for i,c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state==False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),len(good_clients_by_clip_sibyl_attacker),args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end



            cluster_num_dict,cluster_dict = devided_cluster(args,good_clients_by_clip_sibyl_attacker)
            if self.atk_state==False:
                logging.info("cluster_num_dict:"+str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
                    # Manager.client_svd(cluster_dict[key])
                    # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager,cluster_num_dict,cluster_dict)
            if self.atk_state==False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            group_manager.group_mkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx],k=len(clients)//4,largest=True,sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos,incres_cos_idx =torch.topk(Max_Sum,k=int(len(group_manager.byt_group_clients) * 0.20
                                                                ),largest=False,sorted=True)
            byt_in_group = []
            for i,idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx,c.byzantine,incres_cos[i]]))

            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)).difference(set(sibyl_atk)))

            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx,c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state==False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(len(byt_in_group),args.attacker_num,len(group_normal_clients),
                                                                                len(clients), group_state))

            """"""
            ##clip_large and small norm begin
            ##clip_large and small norm end
            ##clip_large and small norm begin
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((group_normal_clients[i].byzantine,group_normal_clients[i].client_idx))
                    byzantine_idx.append(group_normal_clients[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in group_normal_clients:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(group_normal_clients),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))
            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean/norms_min###不能在聚合后补模，会打乱内部梯度的方向
            # global_norm = 1.0
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]/len(normal_clients_clip_by_norm)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict

        if self.args.defend_module == "AD":
            # module AD
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i],dim=0,largest=True,sorted=True,k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos)/sybil_num
            descr_cos,descr_cos_idx = torch.topk(cos_sum_mean,k=sybil_num,sorted=True,largest=True)
            print("cos_sum_mean",cos_sum_mean)
            for i,c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:good_clients_by_clip_sibyl_attacker.append(c)



            "##model_poisoning_defend 3: model aggregation by same norm begin"

            """"""
            ##clip_large and small norm begin
            ##clip_large and small norm end
            ##clip_large and small norm begin
            norms1 = []
            """"""
            for c in good_clients_by_clip_sibyl_attacker:
                norms1.append(c.grad_norm)
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((good_clients_by_clip_sibyl_attacker[i].byzantine,good_clients_by_clip_sibyl_attacker[i].client_idx))
                    byzantine_idx.append(good_clients_by_clip_sibyl_attacker[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in good_clients_by_clip_sibyl_attacker:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(good_clients_by_clip_sibyl_attacker),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))

            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)
            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean/norms_min###不能在聚合后补模，会打乱内部梯度的方向
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]*(1/len(normal_clients_clip_by_norm))
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "AB":
        #module AB
            # if self.atk_state == False:
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i],dim=0,largest=True,sorted=True,k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos)/sybil_num
            descr_cos,descr_cos_idx = torch.topk(cos_sum_mean,k=sybil_num,sorted=True,largest=True)
            print("cos_sum_mean",cos_sum_mean)
            for i,c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state==False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),len(good_clients_by_clip_sibyl_attacker),args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end
            cluster_num_dict,cluster_dict = devided_cluster(args,good_clients_by_clip_sibyl_attacker)
            if self.atk_state==False:
                logging.info("cluster_num_dict:"+str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
                    # Manager.client_svd(cluster_dict[key])
                    # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager,cluster_num_dict,cluster_dict)

            if self.atk_state==False:
                group_manager.show_state(group_manager.group_list)
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            if self.atk_state==False:
                group_mean_vec = []
                Bzt_group_vec = []
                Bn_group_vec = []
                # byt_label = []
                all_group_vec = []
                # for group in group_manager.group_list:
                #     iid_group_vec_local_danwh_grads = []
                #     # all_group_vec.append()
                #     bzt_group_vec = []
                #     bn_group_vec = []
                #     if group.exsiting_byzantine:
                #         # byt_label.append(1)
                #         for c in group.group_clients:
                #             bzt_group_vec.append(c.vec_grad_no_meanvar)
                #         by_vec = torch.mean(torch.stack(bzt_group_vec), dim=0)
                #         Bzt_group_vec.append(by_vec)
                #     else:
                #         # byt_label.append(0)
                #         for c in group.group_clients:
                #             bn_group_vec.append(c.vec_grad_no_meanvar)
                #         bn_vec = torch.mean(torch.stack(bn_group_vec), dim=0)
                #         Bn_group_vec.append(bn_vec)
                #     for c in group.group_clients:
                #         iid_group_vec_local_danwh_grads.append(c.vec_grad_no_meanvar)
                #     vec_avg_grad = torch.mean(torch.stack(iid_group_vec_local_danwh_grads), dim=0).cpu().tolist()
                #     group_mean_vec.append(vec_avg_grad)
                #########shift
                # bn_group_mean_vec = torch.mean(torch.stack(Bn_group_vec), dim=0)
                # by_group_mean_vec = torch.mean(torch.stack(Bzt_group_vec), dim=0)
                # u_shift_gp = torch.norm(bn_group_mean_vec-by_group_mean_vec,2).item()
                # cos_simi_gp = torch.cosine_similarity(bn_group_mean_vec,by_group_mean_vec, 0).item()
                # bzt_clients_vec = []
                # bn_clients_vec = []
                # for c in clients:
                #     if c.byzantine:
                #         bzt_clients_vec.append(c.vec_grad_no_meanvar)
                #     else:
                #         bn_clients_vec.append(c.vec_grad_no_meanvar)
                # bn_mean_vec = torch.mean(torch.stack(bn_clients_vec), dim=0)
                # by_mean_vec = torch.mean(torch.stack(bzt_clients_vec), dim=0)
                # u_shift_cl = torch.norm(bn_mean_vec-by_mean_vec,2).item()
                # cos_simi_cl = torch.cosine_similarity(bn_mean_vec, by_mean_vec, 0).item()
                # wandb.log({"u_shift_gp": u_shift_gp, "round_idx": round_idx})
                # wandb.log({"u_shift_cl": u_shift_cl, "round_idx": round_idx})
                # wandb.log({"cos_simi_gp": cos_simi_gp, "round_idx": round_idx})
                # wandb.log({"cos_simi_cl": cos_simi_cl, "round_idx": round_idx})
                # Max_cos = []
                # Max_dis = []
                # cos_data = []
                # dis_data = []
                # client_to_idx_dict = {}
                # for i, c1 in enumerate(clients):
                #     Distance_among_users = []
                #     cos_among_users = []
                #     dis_among_users = []
                #     for j, c2 in enumerate(clients):
                #         cos_among_users.append(
                #             torch.cosine_similarity(c1.vec_grad_no_meanvar, c2.vec_grad_no_meanvar, 0).item())
                #         dis_among_users.append(torch.norm(c1.vec_grad_no_meanvar - c2.vec_grad_no_meanvar, 2).item())
                #     cos_data.append(cos_among_users)
                #     dis_data.append(dis_among_users)
                #     max_cos, max_idx = torch.topk(torch.tensor(cos_among_users), largest=False, k=3, sorted=True)
                #     max_dis, max_idx = torch.topk(torch.tensor(dis_among_users), largest=True, k=3, sorted=True)
                #     # print(i,max_cos[1],max_idx[1])
                #     Max_cos.append(max_cos[0].item())
                #     Max_dis.append(max_dis[1].item())
                # n = len(clients)
                # c_mean_cos = (np.sum(np.array(cos_data)) - n) / (n * (n - 1))
                # c_mean_dis = 2*np.sum(np.array(dis_data)) / (n * (n - 1))
                # print(round_idx,Max_cos)
                # min_c = np.min(np.array(Max_cos))
                # max_d = np.max(np.array(Max_dis))
                # wandb.log({"client_min_cos": min_c, "round": round_idx})
                # wandb.log({"client_mean_cos": c_mean_cos, "round": round_idx})
                # # wandb.log({"client_max_dis": max_d, "round": round_idx})
                # wandb.log({"client_mean_dis": c_mean_dis, "round": round_idx})
                # Max_cos = []
                # Max_dis = []
                # cos_data = []
                # dis_data = []
                # client_to_idx_dict = {}
                # for i, c1 in enumerate(Bn_group_vec):
                #     Distance_among_users = []
                #     cos_among_users = []
                #     dis_among_users = []
                #     for j, c2 in enumerate(Bn_group_vec):
                #         cos_among_users.append(
                #             torch.cosine_similarity(c1, c2, 0).item())
                #         dis_among_users.append(torch.norm(c1 - c2, 2).item())
                #     cos_data.append(cos_among_users)
                #     dis_data.append(dis_among_users)
                #     max_cos, max_idx = torch.topk(torch.tensor(cos_among_users), largest=False, k=3, sorted=True)
                #     max_dis, max_idx = torch.topk(torch.tensor(dis_among_users), largest=True, k=3, sorted=True)
                #     # print(i,max_cos[1],max_idx[1])
                #     Max_cos.append(max_cos[0].item())
                #     Max_dis.append(max_dis[1].item())
                # n = len(Bn_group_vec)
                # g_mean_cos = (np.sum(np.array(cos_data)) - n) / (n * (n - 1))
                # g_mean_dis = 2*np.sum(np.array(dis_data)) / (n * (n - 1))
                # # print(round_idx,Max_cos)
                # # min_c = np.min(np.array(Max_cos))
                # # max_d = np.max(np.array(Max_dis))
                # # wandb.log({"group_min_cos": min_c, "round": round_idx})
                # wandb.log({"group_mean_cos": g_mean_cos, "round": round_idx})
                # # wandb.log({"group_max_dis": max_d, "round": round_idx})
                # wandb.log({"group_mean_dis": g_mean_dis, "round": round_idx})
                # wandb.log({"cos_mul": (1-c_mean_cos)/(1-g_mean_cos), "round": round_idx})
                # wandb.log({"dis_mul": c_mean_dis / g_mean_dis, "round": round_idx})


            group_manager.group_mkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.n给gormal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx],k=len(clients)//4,largest=True,sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos,incres_cos_idx =torch.topk(Max_Sum,k=int(len(group_manager.byt_group_clients) * 0.20
                                                                ),largest=False,sorted=True)
            byt_in_group = []
            for i,idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx,c.byzantine,incres_cos[i]]))
            # group_normal_clients = list(set(clients))
            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)).difference(set(sibyl_atk)))



            normal_state = []
            selected_byt_num = 0
            for c in group_normal_clients:
                c.weight = 1/len(group_normal_clients)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num":len(group_normal_clients),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, group_normal_clients)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "AC":
            # module ABC
            beta_1 = 0.20  ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients, clients)
            # distance_matric1 = Manager.distance_matric(clients)
            # clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1 * len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk = []
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i], dim=0, largest=True, sorted=True, k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos) / sybil_num
            descr_cos, descr_cos_idx = torch.topk(cos_sum_mean, k=sybil_num, sorted=True, largest=True)
            print("cos_sum_mean", cos_sum_mean)
            for i, c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:
                    good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state == False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(
                    args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),
                    len(good_clients_by_clip_sibyl_attacker), args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end

            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(sibyl_atk)))

            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx, c.byzantine])
                norms1.append(c.grad_norm)

            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss, byt_idx_by_delete_negloss = clip_model_posioned_client(args,
                                                                                                 group_normal_clients,
                                                                                                 self.atk_state,
                                                                                                 round_idx)  # h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(group_normal_clients[idx])
                norms1.append(group_normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(group_normal_clients[idx])
            if self.atk_state == False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss, "Model_Check")

            normal_state = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_loss:
                c.weight = 1 / len(normal_clients_clip_by_loss)
                normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num": len(normal_clients_clip_by_loss), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
                         'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_loss)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "BC":
        #module ABC
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)

            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i

            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            cluster_num_dict,cluster_dict = devided_cluster(args,clients)
            if self.atk_state==False:
                logging.info("cluster_num_dict:"+str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
                    # Manager.client_svd(cluster_dict[key])
                    # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager,cluster_num_dict,cluster_dict)
            if self.atk_state==False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            group_manager.group_mkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx],k=len(clients)//4,largest=True,sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos,incres_cos_idx =torch.topk(Max_Sum,k=int(len(group_manager.byt_group_clients) * 0.20
                                                                ),largest=False,sorted=True)
            byt_in_group = []
            for i,idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx,c.byzantine,incres_cos[i]]))
            # group_normal_clients = list(set(clients))
            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)))

            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx,c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state==False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(len(byt_in_group),args.attacker_num,len(group_normal_clients),
                                                                                len(clients), group_state))
            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss,byt_idx_by_delete_negloss = clip_model_posioned_client(args,group_normal_clients,self.atk_state,round_idx) #h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(group_normal_clients[idx])
                norms1.append(group_normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(group_normal_clients[idx])
            if self.atk_state==False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss,"Model_Check")

            normal_state = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_loss:
                c.weight = 1/len(normal_clients_clip_by_loss)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_loss),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_loss)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "BD":
            # module ABD
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i


            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])

            cluster_num_dict,cluster_dict = devided_cluster(args,clients)
            if self.atk_state==False:
                logging.info("cluster_num_dict:"+str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
                    # Manager.client_svd(cluster_dict[key])
                    # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager,cluster_num_dict,cluster_dict)
            if self.atk_state==False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad()
            group_manager.group_mkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Max_Sum = []
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx],k=len(clients)//4,largest=True,sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos,incres_cos_idx =torch.topk(Max_Sum,k=int(len(group_manager.byt_group_clients) * 0.20
                                                                ),largest=False,sorted=True)
            byt_in_group = []
            for i,idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx,c.byzantine,incres_cos[i]]))

            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)))

            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx,c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state==False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(len(byt_in_group),args.attacker_num,len(group_normal_clients),
                                                                                len(clients), group_state))


            """"""
            ##clip_large and small norm begin
            ##clip_large and small norm end
            ##clip_large and small norm begin
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((group_normal_clients[i].byzantine,group_normal_clients[i].client_idx))
                    byzantine_idx.append(group_normal_clients[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in group_normal_clients:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(group_normal_clients),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))
            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean/norms_min###不能在聚合后补模，会打乱内部梯度的方向
            # global_norm = 1.0
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]*(1/len(normal_clients_clip_by_norm))
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "CD":
            # module ACD
            beta_1 = 0.20 ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients,clients)
            # distance_matric1 = Manager.distance_matric(clients)
            #clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1*len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk =[]
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i



            normal_clients = clients

            #
            # group_state = []
            # """"""
            # norms1 = []
            # """"""
            # for c in group_normal_clients:
            #     group_state.append([c.client_idx,c.byzantine])
            #     norms1.append(c.grad_norm)

            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss,byt_idx_by_delete_negloss = clip_model_posioned_client(args,normal_clients,self.atk_state,round_idx) #h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(normal_clients[idx])
                norms1.append(normal_clients[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(normal_clients[idx])
            if self.atk_state==False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss,"Model_Check")

            "###model_poisoning_defend 2:model_check_end"
            "##model_poisoning_defend 3: model aggregation by same norm begin"

            """"""
            ##clip_large and small norm begin
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((normal_clients_clip_by_loss[i].byzantine,normal_clients_clip_by_loss[i].client_idx))
                    byzantine_idx.append(normal_clients_clip_by_loss[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in normal_clients_clip_by_loss:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(normal_clients_clip_by_loss),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))

            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean/norms_min###不能在聚合后补模，会打乱内部梯度的方向
            # global_norm = 1.0
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]*(1/len(normal_clients_clip_by_norm))
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        # if self.args.defend_module == "B":
        #     # module B
        #     beta_1 = 0.20  ##
        #
        #     metric = {
        #         "attacker_num": 0,
        #         " success rate": 0.}
        #     "##model_poisoning_defend 1_1:devided_group_by_label_begin"
        #     # group_manager = dvided_group(args,normal_clients)
        #     # group_manager.show_group_state()
        #     # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
        #     "##model_poisoning_defend 1_1:devided_group_by_label_end"
        #     "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
        #     self.atk_state = atk_state
        #     normal_clients_clip_by_group = []
        #     Manager = Group_Manager(manager_id=0)
        #     Manager.args = args
        #     Manager.atk_state = self.atk_state
        #     Manager.device = device
        #     cos_matric1 = Manager.cos_matric(clients, clients)
        #     # distance_matric1 = Manager.distance_matric(clients)
        #     # clip_sibyl_attack_by_cos_similarity_begin
        #     # for i,cos_va in enumerate(cos_sum_mean):
        #     #     if cos_va.item() > cos_k:
        #     #         sibyl_atk.append(clients[i])
        #     #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
        #
        #     ##clip_sibyl_attack_by_cos_similarity_end
        #
        #
        #     cluster_num_dict, cluster_dict = devided_cluster(args, clients)
        #     if self.atk_state == False:
        #         logging.info("cluster_num_dict:" + str(cluster_num_dict.values()))
        #     # for key in cluster_dict.keys():
        #     #     if cluster_num_dict[key] != 0:
        #     # Manager.client_svd(cluster_dict[key])
        #     # Manager.client_mkrum(cluster_dict[key])
        #     group_manager = Manager.group_match_by_cluster(Manager, cluster_num_dict, cluster_dict)
        #     if self.atk_state == False:
        #         group_manager.show_group_state()
        #     group_manager.set_group_atk_num()
        #     group_manager.compute_group_vec_avg_grad()
        #     group_manager.group_mkrum(device)
        #     # group_manager.group_svd()
        #     # # ////"""""""
        #     # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
        #     # #////"""""""
        #     Max_Sum = []
        #     cidx_to_clox = {}
        #     for i in range(len(clients)):
        #         cidx_to_clox[clients[i].client_idx] = i
        #     for c in group_manager.byt_group_clients:
        #         c_idx = cidx_to_clox[c.client_idx]
        #         max_cos = torch.topk(cos_matric1[c_idx], k=len(clients) // 4, largest=True, sorted=True)[0]
        #         Max_Sum.append(torch.sum(max_cos))
        #
        #     Max_Sum = torch.stack(Max_Sum)
        #     incres_cos, incres_cos_idx = torch.topk(Max_Sum, k=int(len(group_manager.byt_group_clients) * 0.20
        #                                                            ), largest=False, sorted=True)
        #     byt_in_group = []
        #     for i, idx in enumerate(incres_cos_idx):
        #         idx = int(idx.item())
        #         c = group_manager.byt_group_clients[idx]
        #         byt_in_group.append(c)
        #         logging.info("byt_in_group:{}".format([c.client_idx, c.byzantine, incres_cos[i]]))
        #     # group_normal_clients = list(set(clients))
        #     ##module: A+B+C+D,beta_1 = 0.20
        #     group_normal_clients = list(set(clients).difference(set(byt_in_group)))
        #     if atk_state == False:
        #         update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
        #     group_state = []
        #     """"""
        #     norms1 = []
        #     """"""
        #     for c in group_normal_clients:
        #         group_state.append([c.client_idx, c.byzantine])
        #         norms1.append(c.grad_norm)
        #     if self.atk_state == False:
        #         logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(
        #             len(byt_in_group), args.attacker_num, len(group_normal_clients),
        #             len(clients), group_state))
        #
        #     normal_state = []
        #     selected_byt_num = 0
        #     for c in group_normal_clients:
        #         c.weight = 1 / len(group_normal_clients)
        #         normal_state.append((c.client_idx, c.byzantine))
        #         if c.byzantine: selected_byt_num += 1
        #     if self.atk_state == False:
        #         metric["attacker_num"] = selected_byt_num
        #         wandb.log({"aggregation_total_num": len(group_normal_clients), "round": round_idx})
        #         metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
        #         stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
        #                  'round': round_idx}
        #         logging.info(stats)
        #         wandb.log({"success rate": metric["success rate"], "round": round_idx})
        #     else:
        #         return args.attacker_num, selected_byt_num
        #
        #     global_grad_dict = grad_dict_aggregate(args, device, group_normal_clients)
        #     "##model_poisoning_defend 3: model aggregation by same norm begin"
        #     return global_grad_dict
        if self.args.defend_module == "B":
            # module B
            beta_1 = 0.20  ##

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            dis_matric1 = Manager.dis_matric(clients, clients)
            # distance_matric1 = Manager.distance_matric(clients)
            # clip_sibyl_attack_by_cos_similarity_begin
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])

            ##clip_sibyl_attack_by_cos_similarity_end


            cluster_num_dict, cluster_dict = devided_cluster_by_dist(args, clients)
            if self.atk_state == False:
                logging.info("cluster_num_dict:" + str(cluster_num_dict.values()))
            # for key in cluster_dict.keys():
            #     if cluster_num_dict[key] != 0:
            # Manager.client_svd(cluster_dict[key])
            # Manager.client_mkrum(cluster_dict[key])
            group_manager = Manager.group_match_by_cluster(Manager, cluster_num_dict, cluster_dict)
            if self.atk_state == False:
                group_manager.show_group_state()
            group_manager.set_group_atk_num()
            group_manager.compute_group_vec_avg_grad_one()
            group_manager.group_dismkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Min_Sum = []
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
            for c in group_manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                min_dis = torch.topk(dis_matric1[c_idx], k=len(clients) // 4, largest=False, sorted=True)[0]
                Min_Sum.append(torch.sum(min_dis))

            Min_Sum = torch.stack(Min_Sum)
            incres_cos, incres_cos_idx = torch.topk(Min_Sum, k=int(len(group_manager.byt_group_clients) * 0.20
                                                                   ), largest=True, sorted=True)
            byt_in_group = []
            for i, idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = group_manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx, c.byzantine, incres_cos[i]]))
            # group_normal_clients = list(set(clients))
            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)))
            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx, c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state == False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(
                    len(byt_in_group), args.attacker_num, len(group_normal_clients),
                    len(clients), group_state))

            normal_state = []
            selected_byt_num = 0
            for c in group_normal_clients:
                c.weight = 1 / len(group_normal_clients)
                normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num": len(group_normal_clients), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
                         'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num
            global_grad_dict = grad_dict_aggregate(args, device, group_normal_clients)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "C":
            # module C

            self.atk_state = atk_state
            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_2:devided_group_by_clustering_end"
            "###model_poisoning_defend 2:model_check_begin"
            normal_idx_by_delete_negloss, byt_idx_by_delete_negloss = clip_model_posioned_client(args,
                                                                                                 clients,
                                                                                                 self.atk_state,
                                                                                                 round_idx)  # h获取没有被模型中毒攻击过的用户
            ##module: C = normal_clients_clip_by_loss
            normal_clients_clip_by_loss = []
            byt_clients_clip_by_loss = []
            # norms1 = []
            for idx in normal_idx_by_delete_negloss:
                idx = idx.item()
                normal_clients_clip_by_loss.append(clients[idx])
                # norms1.append(normal_clients_clip_by_loss[idx].grad_norm)
            for idx in byt_idx_by_delete_negloss:
                idx = idx.item()
                byt_clients_clip_by_loss.append(clients[idx])
            if self.atk_state == False:
                update_clip_state(round_idx, normal_clients_clip_by_loss, byt_clients_clip_by_loss, "Model_Check")

            normal_state = []
            selected_byt_num = 0
            for c in normal_clients_clip_by_loss:
                c.weight = 1 / len(normal_clients_clip_by_loss)
                normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num": len(normal_clients_clip_by_loss), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
                         'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_loss)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict

        if self.args.defend_module == "D":
            # module AD


            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"




            "##model_poisoning_defend 3: model aggregation by same norm begin"

            """"""
            ##clip_large and small norm begin
            ##clip_large and small norm end
            ##clip_large and small norm begin
            norms1 = []
            """"""
            for c in clients:
                norms1.append(c.grad_norm)
            norms_median = torch.median(torch.stack(norms1))
            byzantine = []
            byzantine_idx = []
            for i,nm in enumerate(norms1):
                if torch.abs(nm.div(norms_median))>8 or torch.abs(nm.div(norms_median))<0.05:
                    byzantine.append((clients[i].byzantine,clients[i].client_idx))
                    byzantine_idx.append(clients[i].client_idx)
            normal_clients_clip_by_norm = []
            for c in clients:
                if c.client_idx not in byzantine_idx:
                    normal_clients_clip_by_norm.append(c)
            ##clip_large and small norm end
            normal_state = []
            norms = []
            selected_byt_num = 0
            self.atk_state = atk_state

            for c in normal_clients_clip_by_norm:
                c.weight = 1/c.grad_norm
                norms.append(c.grad_norm)
                normal_state.append((c.client_idx,c.byzantine))
                if c.byzantine:selected_byt_num +=1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                # logging.info("norm_testing:del_malicious_norm:{},selected_atknum:{},remaning:{},input_clients_num:{},deleted_malicous_state:{}".format(len(byzantine),selected_byt_num,len(normal_clients_clip_by_norm),len(good_clients_by_clip_sibyl_attacker),byzantine))
                wandb.log({"aggregation_total_num":len(normal_clients_clip_by_norm),"round":round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                print(args.attacker_num, selected_byt_num)
                return args.attacker_num, selected_byt_num
            # norms_min = torch.min(torch.stack(norms, 0))
            norms_mean = torch.mean(torch.stack(norms, 0))

            for c in normal_clients_clip_by_norm:
                c.weight = c.weight.mul(norms_mean)
            global_grad_dict = grad_dict_aggregate(args, device, normal_clients_clip_by_norm)
            # global_norm = norms_mean/norms_min###不能在聚合后补模，会打乱内部梯度的方向
            for key in global_grad_dict.keys():
                global_grad_dict[key] = global_grad_dict[key]*(1/len(normal_clients_clip_by_norm))
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "A":
            # module A
            beta_1 = 0.2  ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients, clients)
            # distance_matric1 = Manager.distance_matric(clients)
            # clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1 * len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk = []
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i], dim=0, largest=True, sorted=True, k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos) / sybil_num
            descr_cos, descr_cos_idx = torch.topk(cos_sum_mean, k=sybil_num, sorted=True, largest=True)
            print("cos_sum_mean", cos_sum_mean)
            for i, c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:
                    good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state == False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(
                    args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),
                    len(good_clients_by_clip_sibyl_attacker), args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end

            # n
            selected_byt_num = 0
            for c in good_clients_by_clip_sibyl_attacker:
                c.weight = 1 / len(good_clients_by_clip_sibyl_attacker)
                # normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num": len(good_clients_by_clip_sibyl_attacker), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
                         'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, good_clients_by_clip_sibyl_attacker)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "AVG":
            # module A
            beta_1 = 0.  ##
            cluster_num = clients[0].class_num

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients, clients)
            # distance_matric1 = Manager.distance_matric(clients)
            # clip_sibyl_attack_by_cos_similarity_begin
            cos = []
            sybil_num = int(beta_1 * len(clients))
            good_clients_by_clip_sibyl_attacker = []
            sibyl_atk = []
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
                sum = torch.sum(torch.topk(cos_matric1[i], dim=0, largest=True, sorted=True, k=sybil_num)[0]).item() - 1
                cos.append(sum)
            cos_sum_mean = torch.tensor(cos) / sybil_num
            descr_cos, descr_cos_idx = torch.topk(cos_sum_mean, k=sybil_num, sorted=True, largest=True)
            print("cos_sum_mean", cos_sum_mean)
            for i, c in enumerate(clients):
                if i in descr_cos_idx:
                    sibyl_atk.append(c)
                else:
                    good_clients_by_clip_sibyl_attacker.append(c)
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])
            if self.atk_state == False:
                update_clip_state(round_idx, good_clients_by_clip_sibyl_attacker, sibyl_atk, "Sibyl_testing")
                logging.info("sibyl_testing:del:{},remaining_clients:{},input_clients:{}".format(
                    args.client_num_per_round - len(good_clients_by_clip_sibyl_attacker),
                    len(good_clients_by_clip_sibyl_attacker), args.client_num_per_round))
            ##clip_sibyl_attack_by_cos_similarity_end

            # n
            selected_byt_num = 0
            for c in good_clients_by_clip_sibyl_attacker:
                c.weight = 1 / len(good_clients_by_clip_sibyl_attacker)
                # normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num": len(good_clients_by_clip_sibyl_attacker), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
                         'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, good_clients_by_clip_sibyl_attacker)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict
        if self.args.defend_module == "RG":
            # module B
            beta_1 = 0.20  ##

            metric = {
                "attacker_num": 0,
                " success rate": 0.}
            "##model_poisoning_defend 1_1:devided_group_by_label_begin"
            # group_manager = dvided_group(args,normal_clients)
            # group_manager.show_group_state()
            # iid_group_danwh_avg_grads = group_manager.compute_group_similarity(device)
            "##model_poisoning_defend 1_1:devided_group_by_label_end"
            "##model_poisoning_defend 1_2:devided_group_by_clustering_begin"
            self.atk_state = atk_state
            normal_clients_clip_by_group = []
            Manager = Group_Manager(manager_id=0)
            Manager.args = args
            Manager.atk_state = self.atk_state
            Manager.device = device
            cos_matric1 = Manager.cos_matric(clients, clients)
            # distance_matric1 = Manager.distance_matric(clients)
            # clip_sibyl_attack_by_cos_similarity_begin
            # for i,cos_va in enumerate(cos_sum_mean):
            #     if cos_va.item() > cos_k:
            #         sibyl_atk.append(clients[i])
            #     else:good_clients_by_clip_sibyl_attacker.append(clients[i])

            ##clip_sibyl_attack_by_cos_similarity_end
            cluster_num = 33
            cluster_num_dict, cluster_dict = devided_cluster(args,clients)
            # if self.atk_state == False:
            logging.info("cluster_num_dict:" + str(cluster_num_dict.values()))


            for cluster in cluster_dict.keys():
                if cluster_num_dict[cluster] !=0:
                    group_id = 0
                    group = Group(group_id, 1, 1)
                    group.with_label = False
                    group.group_clients.extend(cluster_dict[cluster])
                    group_id += 1
                    Manager.update_manager(join_group=group)
            # return Manager
            # group_manager = Manager.group_match_by_cluster(Manager, cluster_num_dict, cluster_dict)
            # Manager.random_group()
            if self.atk_state == False:
                Manager.show_group_state()
            Manager.set_group_atk_num()
            Manager.compute_group_vec_avg_grad()
            Manager.group_mkrum(device)
            # group_manager.group_svd()
            # # ////"""""""
            # group_manager.selected_svd(group_manager.normal_group_clients,group_manager.byt_group_clients)
            # #////"""""""
            Max_Sum = []
            cidx_to_clox = {}
            for i in range(len(clients)):
                cidx_to_clox[clients[i].client_idx] = i
            for c in Manager.byt_group_clients:
                c_idx = cidx_to_clox[c.client_idx]
                max_cos = torch.topk(cos_matric1[c_idx], k=len(clients) // 4, largest=True, sorted=True)[0]
                Max_Sum.append(torch.sum(max_cos))

            Max_Sum = torch.stack(Max_Sum)
            incres_cos, incres_cos_idx = torch.topk(Max_Sum, k=int(len(Manager.byt_group_clients) * 0.20
                                                                   ), largest=False, sorted=True)
            byt_in_group = []
            for i, idx in enumerate(incres_cos_idx):
                idx = int(idx.item())
                c = Manager.byt_group_clients[idx]
                byt_in_group.append(c)
                logging.info("byt_in_group:{}".format([c.client_idx, c.byzantine, incres_cos[i]]))
            # group_normal_clients = list(set(clients))
            ##module: A+B+C+D,beta_1 = 0.20
            group_normal_clients = list(set(clients).difference(set(byt_in_group)))
            if atk_state == False:
                update_clip_state(round_idx, group_normal_clients, byt_in_group, "Group_testing")
            group_state = []
            """"""
            norms1 = []
            """"""
            for c in group_normal_clients:
                group_state.append([c.client_idx, c.byzantine])
                norms1.append(c.grad_norm)
            if self.atk_state == False:
                logging.info("group_testiing:del_byt:{}/{},remaining:{},input_clients_num:{},selected_state:{}".format(
                    len(byt_in_group), args.attacker_num, len(group_normal_clients),
                    len(clients), group_state))

            normal_state = []
            selected_byt_num = 0
            for c in group_normal_clients:
                c.weight = 1 / len(group_normal_clients)
                normal_state.append((c.client_idx, c.byzantine))
                if c.byzantine: selected_byt_num += 1
            if self.atk_state == False:
                metric["attacker_num"] = selected_byt_num
                wandb.log({"aggregation_total_num": len(group_normal_clients), "round": round_idx})
                metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
                stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"],
                         'round': round_idx}
                logging.info(stats)
                wandb.log({"success rate": metric["success rate"], "round": round_idx})
            else:
                return args.attacker_num, selected_byt_num

            global_grad_dict = grad_dict_aggregate(args, device, group_normal_clients)
            "##model_poisoning_defend 3: model aggregation by same norm begin"
            return global_grad_dict

    def MAB_FL(self, round_idx,clients,atk_state,cur_model,args,device,model_trainer):
        # clip
        selectedId = [clients[i].client_idx for i in range(len(clients))]
        print("round_{}_selectedId:{}".format(round_idx,selectedId))#d对应的dict
        client_idex_to_client = {c.client_idx:c for c in clients}
        for c in clients:  # 把所有被选中的用户的动量单位化
            # print(c.vec_grad_one[:5])
            if round_idx == 0:
                c.momentum = c.vec_grad_no_meanvar
            else:
                c.momentum =c.vec_grad_no_meanvar + c.miu * (round_idx - c.seleted_epoch) * c.momentum
            c.momentum = c.momentum /torch.norm(c.momentum,p =2)
            # c.seleted_epoch = round_idx
            # c.test_num =1+ (round_idx+1)*c.test_num
        '''
        # 剔除恶意用户/剔除女巫
        G = nx.Graph()
        edges = []
        cos_similarity = []
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                cos_similarity.append(torch.cosine_similarity(clients[i].momentum, clients[j].momentum, dim =0))
                if torch.cosine_similarity(clients[i].momentum, clients[j].momentum, dim =0) > 0.8:
                    edges.append((clients[i].client_idx, clients[j].client_idx))
        G.add_nodes_from(selectedId)
        G.add_edges_from(edges)
        C = sorted(nx.connected_components(G), key=len, reverse=False)
        # print("cos_similarity",cos_similarity)
        print("The graphs:",C)
        print("The minimum graph{}:".format(C[0],C[0]))  # 去除最小的连通子图上的点
        if (len(C[0]) > 1):
            for i in C[0]:
                # client_idex_to_client[i].fail += 1
                selectedId.remove(i)
        print("sybil removal: remaining:",selectedId)
        '''
        local_updates = np.array(torch.stack([client_idex_to_client[client_idx].momentum for client_idx in selectedId],0).cpu().tolist())
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(local_updates)

        # recon = pca.inverse_transform(X_reduced)
        # rmse = [mean_squared_error(local_updates[i], recon[i], squared=False) for i in range(len(local_updates))]
        # print("X_reduced:",X_reduced)

        estimator = SpectralClustering(n_clusters=2)  # 构造聚类器
        estimator.fit(X_reduced)  # 聚类
        # center = estimator.cluster_centers_
        # print("center[0]",center[0])
        # print("center[1]",center[1])
        # center_dist = np.linalg.norm(center[0] - center[1])
        # print("center_dist:", center_dist)
        label_pred = estimator.labels_  # 获取聚类标签
        selectedId_c1 = []
        selectedId_c2 = []
        for i in range(len(selectedId)):
            if label_pred[i] == 0:
                selectedId_c1.append(selectedId[i])
            else:
                selectedId_c2.append(selectedId[i])
        print("selectedId_c1",selectedId_c1)
        print("selectedId_c2",selectedId_c2)
        print("label_pred",label_pred)

        m1 = torch.mean(torch.stack([client_idex_to_client[i].momentum for i in selectedId_c1], axis=0))
        m2 = torch.mean(torch.stack([client_idex_to_client[i].momentum for i in selectedId_c2], axis=0))
        cos_between_clusters = torch.cosine_similarity(m1, m2, 0).item()
        print("cos_between_clusters(-0.1):",cos_between_clusters)
        if cos_between_clusters < 0.1:  # 不相似，选大类
            if len(selectedId_c1) > len(selectedId_c2):
                for i in selectedId_c2:
                    # client_idex_to_client[i].fail += 1  # 被剔除，就记录
                    selectedId.remove(i)
            else:
                for i in selectedId_c1:
                    # client_idex_to_client[i].fail += 1
                    selectedId.remove(i)
        # for i in selectedId:  # 相似，选所有
            # client_idex_to_client[i].succ += 1  # 上传正常更新的次数
        print("final aggregation:", selectedId)
        print([client_idex_to_client[j].grad_norm for j in selectedId])
        norm = torch.stack([client_idex_to_client[i].grad_norm for i in selectedId])
        print("norm",norm)
        lr = torch.median(norm)
        print("lr",lr)
        normal_clients = []
        norm_grad_vec = []
        for c in clients:
            if c.client_idx in selectedId:
                normal_clients.append(c)
                norm_grad_vec.append(c.vec_grad_one)
                # clients.remove(c)
        grad_mean = torch.mean(torch.stack(norm_grad_vec),0).cpu().tolist()
        # print(grad_mean[:2])
        global_grad_dict = collections.OrderedDict()
        for key,value in clients[0].grad.items():
            temp = grad_mean[:value.numel()]
            global_grad_dict[key] = torch.tensor(temp).view(value.size()).to(device).mul(lr)
            del grad_mean[:value.numel()]
        metric = {
            "attacker_num": 0,
            " success rate": 0.}
        for i, c in enumerate(normal_clients):
            if c.byzantine:    metric["attacker_num"] +=1
        if atk_state :return args.attacker_num,metric["attacker_num"]
        else:
            metric["success rate"] = 1 - metric["attacker_num"] / (args.attacker_num + 0.00001)
            stats = {'success rate': metric["success rate"], 'attacker_num': metric["attacker_num"], 'round': round_idx}
            logging.info(stats)
            wandb.log({"success rate": metric["success rate"], "round": round_idx})

        return normal_clients,global_grad_dict

'''
    def MAB_2(self, round_idx,clients,atk_state,cur_model,args,device,model_trainer):

        selectedId = [clients[i].client_idx for i in range(len(clients))]
        print("round_{}   selectedId:{}".format(round_idx, selectedId))

        client_idex_to_client = {c.client_idx: c for c in clients}
        for c in clients:  # 把所有被选中的用户的动量单位化
            # print(c.vec_grad_one[:5])
            if round_idx == 0:
                c.momentum = c.vec_grad_no_meanvar
            else:
                c.momentum = c.vec_grad_no_meanvar + c.miu * (round_idx - c.seleted_epoch) * c.momentum
            c.momentum = c.momentum / torch.norm(c.momentum, p=2)


'''



