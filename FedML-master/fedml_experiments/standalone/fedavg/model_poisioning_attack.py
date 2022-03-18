import copy
import logging
import random
import argparse
import numpy as np
import torch
import wandb
import collections
import math
from scipy.stats import norm
import os
from fedml_api.standalone.fedavg.client import Client
from fedml_core.robustness.robust_aggregation import RobustAggregator
from fedml_core.robustness import robust_aggregation as RAG



class Model_Attacker(object):
    def __init__(self,round_idx,args,device,model_trainer,arggregator,aided_data_test,w_global):
        self.w_global = w_global
        self.round_idx = round_idx
        self.arggregator = arggregator
        self.model_trainer = model_trainer
        self.args = args
        self.device = device
        self.arggregator = arggregator
        self.global_loss = None
        self.test_global = None
        self.test_global_aided = None
        self._aided_data_test = aided_data_test
        self.atk_state = False
        self.pre_obj = 0.8

        if self.args.defend_module== "AVG":
            self.fenzi =0.4
        else:
            self.fenzi = 0.4
        self.max_distance_atk = ["median", "fedavg", "trimmed","fltrust"]
        self.max_num_atk = ["MKrum","dnc","fedbt","Zeno"]
        self.min_max_atk = ["min_max_updates_only","min_max_agnostic"]
        self.min_sum_atk = ["min_sum_updates_only","min_sum_agnostic"]
        self.atk_knowlege = ["arg_and_update","arg_only"]
        self.atk_knowlege_no_arg = ["min_max_agnostic","min_sum_agnostic","min_max_updates_only","min_sum_updates_only"]
        self.vec_fedavg = None
        self.deta_p = None
        self.r = 0
        self.achive_obj_flag = False
        self.fedavg_grad_dict = None
        self.pre_r = 0
        self.max_sum_distance = 0
        self.max_distance = 0


    def set_deta_p(self,clients):
        # resnet20 317848
        # self.deta_p = -self.vec_fedavg.div(torch.norm(self.vec_fedavg,2))##r = 200
        # self.deta_p = -torch.sign(self.vec_fedavg)
        vec_grad = []
        for c in clients:
            vec_grad.append(c.vec_grad_one)
        self.deta_p = -torch.sign(torch.std(torch.stack(vec_grad),0))#0.0006

    def distance_matric(self,clients):
        matric = []
        for ci in clients:
            d_ci_to_all = []
            for cj in clients:
                d_ci_to_all.append(torch.norm(RAG.vectorize_state_dict(ci.grad, self.device) - RAG.vectorize_state_dict(cj.grad,self.device),2))
            matric.append(torch.stack(d_ci_to_all))
        return torch.stack(matric)

    def compute_max_distance_between_clients(self,clients):
        matric = self.distance_matric(clients)
        max_in_row = torch.max(matric,dim=1)[0]
        return torch.max(max_in_row).item()

    def compute_max_sum_of_distance_between_clients(self,clients):
        matric = self.distance_matric(clients)
        sum_in_row = torch.sum(matric,dim=1)
        return torch.max(sum_in_row).item()

    def optimzie_attack_object(self,clients):

        if self.args.attacker_knowlege in self.atk_knowlege:
            if self.args.defend_type in self.max_distance_atk:
                roubust_avg_dict = self.robust_result(clients)
                vec_roubust_avg = RAG.vectorize_state_dict(roubust_avg_dict, self.device)
                cur_obj = torch.norm(self.vec_fedavg - vec_roubust_avg, 2).item()
                logging.info("---optimize obj...", )
                logging.info("-obj_change>>distance:{} -- >{}".format(self.pre_obj, cur_obj))
                self.pre_obj = cur_obj
                if self.pre_obj <= cur_obj:
                    return True

                else:
                    return False
            if self.args.defend_type in self.max_num_atk:
                all_atk_num,select_atk_num = self.robust_result(clients)
                logging.info("---optimize obj...")
                logging.info("-obj_change>>attack_num:{} -- >{}/{}".format(self.pre_obj,select_atk_num,all_atk_num))
                self.pre_obj = select_atk_num
                if select_atk_num == all_atk_num:
                    # self.pre_obj = select_atk_num
                    return True
                else:
                    # self.pre_obj = select_atk_num
                    return False
        else:
            if self.args.attacker_knowlege in self.min_max_atk:
                distance  = self.distance_matric(clients)
                for i,c in enumerate(clients):
                    if c.byzantine:
                        cur_obj = torch.max(distance[i]).item()
                        break
                logging.info("---optimize min_max_obj...")
                logging.info("-obj_change>>min_max_distance:{} -- >{}/{}".format(self.pre_obj,cur_obj,self.max_distance))
                self.pre_obj = cur_obj
                if cur_obj == self.max_distance:return True
                else:return False

            if self.args.attacker_knowlege in self.min_sum_atk:
                distance  = self.distance_matric(clients)
                for i,c in enumerate(clients):
                    if c.byzantine:
                        cur_obj = torch.sum(distance[i]).item()
                        break
                logging.info("---optimize min_sum_obj...")
                logging.info("-obj_change>>min_max_sum_of_distance:{} -- >{}/{}".format(self.pre_obj, cur_obj,self.max_sum_distance))
                self.pre_obj = cur_obj
                if cur_obj == self.max_sum_distance :
                    return True
                else:
                    return False

    def robust_result(self,clients):

        if self.args.defend_type == "fedavg":
            avg_dict = self.arggregator.fedavg(self.round_idx, clients, self.w_global, self.args, self.device,
                              )
        if self.args.defend_type == "fedbt":
            avg_dict = self.arggregator.fedbt(self.round_idx, clients, self.atk_state, self.w_global,self.args,
                                              self.device,self.model_trainer
                                              )
        if self.args.defend_type == "MKrum":
            avg_dict = self.arggregator.MKrum(self.round_idx,self.atk_state,clients,self.args,self.device)
        if self.args.defend_type == "dnc":
            avg_dict = self.arggregator.dnc(self.round_idx, clients, self.atk_state,
                                                    self.args, self.device)
        if self.args.defend_type == "Zeno":
            avg_dict = self.arggregator.Zeno(self.round_idx, clients, self.w_global, self.test_global,
                                                  self.args, self.device, self.model_trainer,self.atk_state)
        if self.args.defend_type == "median":
            avg_dict = self.arggregator.median(self.round_idx, self.w_global, clients, self.args, self.device, self.model_trainer)
        if self.args.defend_type == "fltrust":
            avg_dict = self.arggregator.fltrust(self.round_idx, clients, self.w_global, self.test_global,
                                                  self.args, self.device, self.model_trainer)
        if self.args.defend_type == "trimmed":
            avg_dict = self.arggregator.trimmed(self.round_idx,clients,self.w_global,self.test_global,self.args,self.device,self.model_trainer)
        return avg_dict

    def test_model(self,round_idx,client,w_global):
        model = collections.OrderedDict()
        for key, name in w_global.items():
            # if RAG.is_weight_param(key):
                model[key] = w_global[key].to(self.device) - client.grad[key].to(self.device)
            # else:model[key] = client.grad[key].to(self.device)
        # client.model_state_dict = model
        client.test = 2
        self.model_trainer.set_model_params(model)
        stats = self._aided_data_test(round_idx, client)
        # global_loss = 50
        # print("self.global_loss",self.global_loss)
        # print("aided_test_loss",stats["aided_test_loss"])
        client.aid_loss = self.global_loss - stats["aided_test_loss"]
        self.model_trainer.set_model_params(w_global)

    def opt_r(self,clients):
        r = self.args.init_r
        self.pre_r = r
        pre_pre_obj = 0
        tao = 0.000000000001
        r_succ = 0
        step = r/2
        pre_step = step
        flag = True
        print("r", r_succ)
        self.fedavg_grad_dict = self.arggregator.fedavg(self.round_idx, clients, copy.deepcopy(self.w_global),
                                                self.args, self.device)  # 按模型加权聚合
        self.vec_fedavg = RAG.vectorize_state_dict(self.fedavg_grad_dict,self.device)
        self.set_deta_p(clients)
        # if self.args.defend_type in self.max_distance_atk:
        #     self.pre_obj = torch.norm(self.vec_fedavg, 2)
        #     print("self.pre_obj",self.pre_obj)
        while (np.abs(r_succ - r) > tao) or flag:
            flag =False
            STATE = self.optimize_obj(clients, r)
            if STATE:
                r_succ = r
                logging.info("true_r"+str(r)+"step"+str(step)+"r_succ"+str(r_succ))
                r += step / 2
            else:
                r -= step / 2
                logging.info("false_r"+str(r)+"step"+str(step)+"r_succ"+str(r_succ))
            step = step / 2
            logging.info("pre_r:"+str(self.pre_r)+"r:"+str(r)+"pre_obj:"+str(self.pre_obj)+"pre_pre_obj:"+str(pre_pre_obj))
            if (self.args.attacker_knowlege in self.atk_knowlege_no_arg) or (self.args.defend_type in self.max_distance_atk):
                if self.args.init_r < 1:
                    if np.abs(pre_step - step)<0.1 or abs(self.pre_obj - pre_pre_obj) < 0.01:break

                    else:
                        self.pre_r = r
                        pre_pre_obj = self.pre_obj
                else:
                    if np.abs(pre_step - step)<0.1 or abs(self.pre_obj - pre_pre_obj) < 0.1:break
                    else:
                        self.pre_r = r
                        pre_pre_obj = self.pre_obj
            else:
                if self.args.init_r < 1:
                    if np.abs(pre_step - step)<0.01:break
                    else:
                        self.pre_r = r
                        pre_pre_obj = self.pre_obj
                        pre_step = step
                else:
                    if np.abs(self.pre_r - r)<1:break
                    else:
                        self.pre_r = r
                        pre_pre_obj = self.pre_obj
                        pre_step = step
        return r_succ

    def optimize_obj(self,clients, r):
        logging.info("******dnc_atk...")
        logging.info("---r: "+str(r))
        self.r = r
        # print("grad_norm",grad_norm)
        logging.info("---setting poisonsing model..")
        pre_loss = []
        for c in clients:
            pre_loss.append([c.byzantine,c.aid_loss])
        for i,c in enumerate(clients):
            if self.args.attacker_knowlege in self.atk_knowlege_no_arg:
                if c.byzantine:
                    byt_m = self.vec_fedavg.add(self.r * self.deta_p).cpu().numpy().tolist()  ##添加扰动
                    poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                    vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                    poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                    c.vec_grad_one = vec_poisoned_grad_one
                    c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad, self.device)
                    c.grad = poisoned_grad
                    c.grad_norm = poisoned_grad_norm
                    if self.args.defend_type == "fedbt":
                        self.test_model(self.round_idx,c,copy.deepcopy(self.w_global))
                    break
            else:
                if c.byzantine:
                    if i == 0:
                        deta_p = self.deta_p
                    else:
                        if self.args.defend_type == "fedbt" or "fedavg":
                            # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                            self.loc_num = int(self.deta_p.numel() * self.fenzi)
                            sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num,
                                                          replace=False).tolist()

                            deta_p = copy.deepcopy(self.deta_p)
                            deta_p[sample_loc] = -deta_p[sample_loc]
                        else:
                            if (self.args.defend_type == "fltrust" or "d")  and self.args.model== "lr":
                                self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                np.random.seed(1)
                                sample_loc = np.random.choice(range(self.deta_p.numel()), int(self.loc_num),
                                                              replace=False).tolist()

                                deta_p = copy.deepcopy(self.deta_p)
                                deta_p[sample_loc] = -deta_p[sample_loc]
                            else:
                                deta_p = self.deta_p
                    byt_m = self.vec_fedavg.add(self.r * deta_p).cpu().numpy().tolist()  ##添加扰动
                    poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                    vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                    poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                    c.vec_grad_one = vec_poisoned_grad_one
                    c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad, self.device)
                    c.grad = poisoned_grad
                    c.grad_norm = poisoned_grad_norm
                    if self.args.defend_type == "fedbt":
                        self.test_model(self.round_idx,c,copy.deepcopy(self.w_global))
        result = self.optimzie_attack_object(clients)
        # print("---loss_change")
        # print("-curloss",loss)
        logging.info("******dnc_atk_end...")
        return result

    def fedbt_tailored_attack(self,clients):
        r_succ = self.opt_r(clients)
        return r_succ

        # print("r_succ", r_succ)
    def sign_flipping(self,clients):
        print("attack_type", self.args.attacker_knowlege)
        for c in clients:
            if c.byzantine:
                for (name,_) in c.grad.items():
                    c.grad[name] = c.grad[name].mul(-5)
                c.vec_grad_one = c.vec_grad_one.mul(-1)

    def random_grad(self,clients):
        if self.args.attack_type == "random_grad":
            print("self.args.attack_type", self.args.attacker_knowlege)
            for c in clients:
                if c.byzantine:
                    for (name, parm) in c.grad.items():
                        c.grad[name] = torch.normal(mean=0,std=2,size=parm.size()).div(1e20)
                    c.grad_one = c.grad_one.mul(c.grad_norm).div(1e20)

    def dnc_attack(self,clients):
        if self.args.attack_type == "model_attack":
            new_clients = []
            if self.args.attacker_knowlege == "arg_and_update":#agr-updatesadversary is the strongest adversary who knows both the gradients of benign devices and the server’s AGR.
                logging.info("arg_and_update")
                self.atk_state = True
                r_succ = self.fedbt_tailored_attack(clients)
                print("ddd")
                logging.info("r_succ:{}_cur_r{}".format(r_succ,self.r))
                for i,c in enumerate(clients):
                    if c.byzantine:
                        if i == 0:deta_p = self.deta_p
                        else:
                            if self.args.defend_type == "fedbt" or "fedavg":
                                # if self.args.client_num_per_round == 60:
                                #     deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                # else:
                                    # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                    self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num,
                                                                  replace=False).tolist()
                                    # print("sample_loc:",sample_loc[:5])
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -deta_p[sample_loc]
                                    # deta_p = copy.deepcopy(self.deta_p)
                            else:
                                if self.args.defend_type == "dnc" and self.args.model == "lr":
                                    self.loc_num = self.deta_p.numel() // 2
                                    np.random.seed(1)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()), int(self.loc_num),
                                                                  replace=False).tolist()
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -self.deta_p[sample_loc]
                                else:deta_p = self.deta_p
                        byt_m = self.vec_fedavg.add(self.r* deta_p).cpu().numpy().tolist()  ##添加扰动
                        poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                        vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                        poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                        # model_state_dict = collections.OrderedDict()
                        # for key in self.fedavg_grad_dict.keys():
                        #     model_state_dict[key] = self.w_global[key].to(self.device) - poisoned_grad[key].to(
                        #         self.device)
                        # c.model_state_dict = model_state_dict
                        c.vec_grad_one = vec_poisoned_grad_one
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        if self.args.defend_type == "fedbt":
                            self.test_model(self.round_idx, c, copy.deepcopy(self.w_global))
                    new_clients.append(c)
                self.atk_state = False

            if self.args.attacker_knowlege == "arg_only":#agr-only adversary knows the server’s AGR, but does not have the gradients of benign devices.
                logging.info("arg_only")
                self.atk_state = True
                malicious_clients = []
                for c in clients:
                    if c.byzantine:
                        malicious_clients.append(c)
                r_succ = self.fedbt_tailored_attack(malicious_clients)
                logging.info("r_succ:{}_cur_r{}".format(r_succ,self.r))
                for i,c in enumerate(clients):
                    if c.byzantine:
                        if i == 0:deta_p = self.deta_p
                        else:
                            if self.args.defend_type == "fedbt":
                                if self.args.client_num_per_round == 60:
                                    deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                else:
                                    # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                    self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num,
                                                                  replace=False).tolist()
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -deta_p[sample_loc]
                            else:deta_p = self.deta_p
                        byt_m = self.vec_fedavg.add(self.r* deta_p).cpu().numpy().tolist()  ##添加扰动
                        poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                        vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                        poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                        # model_state_dict = collections.OrderedDict()
                        # for key in self.fedavg_grad_dict.keys():
                        #     model_state_dict[key] = self.w_global[key].to(self.device) - poisoned_grad[key].to(
                        #         self.device)
                        # c.model_state_dict = model_state_dict
                        c.vec_grad_one = vec_poisoned_grad_one
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        if self.args.defend_type == "fedbt":
                            self.test_model(self.round_idx, c, copy.deepcopy(self.w_global))
                    new_clients.append(c)
                self.atk_state = False

            if self.args.attacker_knowlege == "min_max_updates_only":#updates-only adversary has the gradients of benign devices, but does not know the server’s AGR.
                logging.info("min_max_updates_only")
                self.max_distance = self.compute_max_distance_between_clients(clients)
                r_succ = self.fedbt_tailored_attack(clients)
                logging.info("r_succ:{}_cur_r{}".format(r_succ,self.r))
                for i,c in enumerate(clients):
                    if c.byzantine:
                        if i == 0:deta_p = self.deta_p
                        else:
                            if self.args.defend_type == "fedbt" or "fedavg":
                                # if self.args.client_num_per_round == 60:
                                #     deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                # else:
                                    # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                    self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num ,
                                                                  replace=False).tolist()
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -deta_p[sample_loc]
                            else:deta_p = self.deta_p
                        byt_m = self.vec_fedavg.add(self.r* deta_p).cpu().numpy().tolist()  ##添加扰动
                        poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                        vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                        poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                        # model_state_dict = collections.OrderedDict()
                        # for key in self.fedavg_grad_dict.keys():
                        #     model_state_dict[key] = self.w_global[key].to(self.device) - poisoned_grad[key].to(
                        #         self.device)
                        # c.model_state_dict = model_state_dict
                        c.vec_grad_one = vec_poisoned_grad_one
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        if self.args.defend_type == "fedbt":
                            self.test_model(self.round_idx, c, copy.deepcopy(self.w_global))
                    new_clients.append(c)

            if self.args.attacker_knowlege == "min_max_agnostic":#updates-only adversary has the gradients of benign devices, but does not know the server’s AGR.
                logging.info("min_max_agnostic")
                malicious_clients = []
                for c in clients:
                    if c.byzantine:
                        malicious_clients.append(c)
                self.max_distance = self.compute_max_distance_between_clients(malicious_clients)
                r_succ = self.fedbt_tailored_attack(malicious_clients)
                logging.info("r_succ:{}_cur_r{}".format(r_succ,self.r))
                for i,c in enumerate(clients):
                    if c.byzantine:
                        if i == 0:deta_p = self.deta_p
                        else:
                            if self.args.defend_type == "fedbt":
                                if self.args.client_num_per_round == 60:
                                    deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                else:
                                    # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                    self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num ,
                                                                  replace=False).tolist()
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -deta_p[sample_loc]
                            else:deta_p = self.deta_p
                        byt_m = self.vec_fedavg.add(self.r* deta_p).cpu().numpy().tolist()  ##添加扰动
                        poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                        vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                        poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                        # model_state_dict = collections.OrderedDict()
                        # for key in self.fedavg_grad_dict.keys():
                        #     model_state_dict[key] = self.w_global[key].to(self.device) - poisoned_grad[key].to(
                        #         self.device)
                        # c.model_state_dict = model_state_dict
                        c.vec_grad_one = vec_poisoned_grad_one
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        if self.args.defend_type == "fedbt":
                            self.test_model(self.round_idx, c, copy.deepcopy(self.w_global))
                    new_clients.append(c)

            if self.args.attacker_knowlege == "min_sum_updates_only": #updates-only adversary has the gradients of benign devices, but does not know the server’s AGR.
                logging.info("min_sum_updates_only")
                self.max_sum_distance = self.compute_max_sum_of_distance_between_clients(clients)
                r_succ = self.fedbt_tailored_attack(clients)
                logging.info("r_succ:{}_cur_r{}".format(r_succ,self.r))

                for i,c in enumerate(clients):
                    if c.byzantine:
                        if i == 0:deta_p = self.deta_p
                        else:
                            if self.args.defend_type == "fedbt" or "fedavg":
                                # if self.args.client_num_per_round == 60:
                                #     deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                # else:
                                    # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                    self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num ,
                                                                  replace=False).tolist()
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -deta_p[sample_loc]
                            else:deta_p = self.deta_p
                        byt_m = self.vec_fedavg.add(self.r * deta_p).cpu().numpy().tolist()  ##添加扰动
                        poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                        vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                        poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                        # model_state_dict = collections.OrderedDict()
                        # for key in self.fedavg_grad_dict.keys():
                        #     model_state_dict[key] = self.w_global[key].to(self.device) - poisoned_grad[key].to(
                        #         self.device)
                        # c.model_state_dict = model_state_dict
                        c.vec_grad_one = vec_poisoned_grad_one
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        if self.args.defend_type == "fedbt":
                            self.test_model(self.round_idx, c, copy.deepcopy(self.w_global))
                    new_clients.append(c)

            if self.args.attacker_knowlege == "min_sum_agnostic":#updates-only adversary has the gradients of benign devices, but does not know the server’s AGR.
                logging.info("min_sum_agnostic")
                malicious_clients = []
                for c in clients:
                    if c.byzantine:
                        malicious_clients.append(c)
                self.max_sum_distance = self.compute_max_sum_of_distance_between_clients(malicious_clients)
                r_succ = self.fedbt_tailored_attack(malicious_clients)
                logging.info("r_succ:{}_cur_r{}".format(r_succ,self.r))
                for i,c in enumerate(clients):
                    if c.byzantine:
                        if i == 0:deta_p = self.deta_p
                        else:
                            if self.args.defend_type == "fedbt":
                                if self.args.client_num_per_round == 60:
                                    deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                else:
                                    # deta_p = torch.sign(torch.normal(0, 1, size=(self.deta_p.size())).to(self.device))
                                    self.loc_num = int(self.deta_p.numel() * self.fenzi)
                                    sample_loc = np.random.choice(range(self.deta_p.numel()),self.loc_num ,
                                                                  replace=False).tolist()
                                    deta_p = copy.deepcopy(self.deta_p)
                                    deta_p[sample_loc] = -deta_p[sample_loc]
                            else:deta_p = self.deta_p
                        byt_m = self.vec_fedavg.add(self.r* deta_p).cpu().numpy().tolist()  ##添加扰动
                        poisoned_grad = RAG.recover_to_dict(byt_m, self.fedavg_grad_dict, self.device)
                        vec_poisoned_grad_one = RAG.vectorize_state_dict(poisoned_grad, self.device)
                        poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                        # model_state_dict = collections.OrderedDict()
                        # for key in self.fedavg_grad_dict.keys():
                        #     model_state_dict[key] = self.w_global[key].to(self.device) - poisoned_grad[key].to(
                        #         self.device)
                        # c.model_state_dict = model_state_dict
                        c.vec_grad_one = vec_poisoned_grad_one
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        if self.args.defend_type == "fedbt":
                            self.test_model(self.round_idx, c, copy.deepcopy(self.w_global))
                    new_clients.append(c)

            return new_clients
    def a_little_attack(self,clients):
        if self.args.attack_type == "model_attack":
            if self.args.attacker_knowlege == "a_little":
                #create a little attack model
                s = math.floor(self.args.client_num_per_round / 2 + 1) - self.args.attacker_num
                pro = norm.ppf((self.args.client_num_per_round-s)/self.args.client_num_per_round)
                print("s: ",s)
                print("pro: ",pro)
                vec_grad = []
                for c in clients:
                    vec_grad.append(c.vec_grad_one)
                    print(c.vec_grad_one.shape)
                uv = torch.mean(torch.stack(vec_grad),0)
                print("uv",uv[:10])
                std = torch.std(torch.stack(vec_grad),0)#0.0006
                print("uv*std: ", pro*std[:10])
                poisoned_vec_grad = uv + pro*std

                vec_grad_list = poisoned_vec_grad.cpu().numpy().tolist()
                poisoned_grad = collections.OrderedDict()
                # print("poisoned_vec_grad_norm:", torch.norm(poisoned_vec_grad,2))
                for na, param in self.w_global.items():
                    gp = vec_grad_list[:param.numel()]
                    poisoned_grad[na] = torch.tensor(gp).to(self.device).view(param.size())
                for i,c in enumerate(clients):
                    if c.byzantine:
                        poisoned_grad_norm = torch.norm(poisoned_vec_grad, 2)
                        c.vec_grad_one = poisoned_vec_grad
                        c.vec_grad_no_meanvar = RAG.vectorize_weight(poisoned_grad,self.device)
                        c.grad = poisoned_grad
                        c.grad_norm = poisoned_grad_norm
                        # if i==1:
                        # print("grad_norm:",c.grad_norm)
        return clients

    def PESS_attack(self,atk,clients):
        new_clients = []
        # atk_seting = {l}
        self.atk_state = False
        avg_dict = self.arggregator.fedavg(self.round_idx, clients, copy.deepcopy(self.w_global), self.args,
                                           self.device)
        for client in clients:
            if client.byzantine:
                for key, va in avg_dict.items():
                    client.model[key] = self.w_global[key].to(self.device) - avg_dict[key].to(self.device)
                    # print("vv")
                client.grad = self.model_trainer.PESS_attack_train(self.round_idx,self.args, self.device,atk,client,clients)
                vec_poisoned_grad_one = RAG.vectorize_state_dict(client.grad, self.device)
                poisoned_grad_norm = torch.norm(vec_poisoned_grad_one, 2)
                client.vec_grad_one = vec_poisoned_grad_one
                client.vec_grad_no_meanvar = RAG.vectorize_weight(client.grad, self.device)
                client.grad_norm = poisoned_grad_norm
            new_clients.append(client)

        self.atk_state = False
        return new_clients





