import copy
import logging
import random
import argparse
import numpy as np
import torch
import wandb
import collections
import os
from fedml_experiments.standalone.fedavg.client import Client
from fedml_core.robustness.robust_aggregation import RobustAggregator
from fedml_core.robustness import robust_aggregation as RAG
from fedml_experiments.standalone.fedavg.model_poisioning_attack import Model_Attacker

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, train_labels_type_dic, test_data_local_dict,
         aided_dataset, global_test_aided_data, class_num]=dataset
        self.atk_knowlege = ["min_max_updates_only","min_max_agnostic","arg_and_update","arg_only","min_sum_updates_only","min_sum_agnostic"]
        self.train_global = train_data_global
        self.test_global = test_data_global
        # self.targetted_task_test_loader = targetted_task_test_loader
        # self.poisoned_train_loader = poisoned_train_loader
        # self.num_dps_poisoned_dataset = num_dps_poisoned_dataset
        self.global_test_aided_data = global_test_aided_data
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.train_labels_type_dic = train_labels_type_dic
        self.test_data_local_dict = test_data_local_dict
        self.auxiliary_data_dict = aided_dataset
        self.model_trainer = model_trainer
        self.global_loss = 0
        self.class_num = class_num
        self.model_dim = 0
        self.client_selected_record_dict = {i:0 for i in range(self.args.client_num_in_total)}
        self.client_succ = {i:1 for i in range(self.args.client_num_in_total)}
        self.client_fail = {i:1 for i in range(self.args.client_num_in_total)}
        self.client_momun = {i:None for i in range(self.args.client_num_in_total)}


        self._setup_clients(train_data_local_num_dict, train_data_local_dict,train_labels_type_dic, test_data_local_dict, model_trainer,aided_dataset)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict,train_labels_type_dic, test_data_local_dict, model_trainer,aided_data):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx],train_labels_type_dic[client_idx], self.args, self.device, model_trainer,aided_data[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def get_local_grad(self, args, device, cur_params, init_params):
        update_dict = collections.OrderedDict()
        for k in cur_params.keys():
            # if self.is_weight_param(k):
            update_dict[k] = init_params[k].to(device) - cur_params[k].to(device)
            # else:update_dict[k] = cur_params[k].to(device)
        return update_dict
    def danweihua(self,grad, device):
        vec = RAG.vectorize_state_dict(grad,device)
        one_vec = vec / torch.norm(vec, 2)
        return one_vec
    def test_model(self,round_idx,client,w_global):
        model = collections.OrderedDict()
        for key, param in w_global.items():
            # if RAG.is_weight_param(key):
            model[key] = w_global[key].to(self.device) - client.grad[key].to(self.device)
            # else:model[key] = client.grad[key].to(self.device)
        self.model_trainer.set_model_params(model)
        # client.model_state_dict = model
        # self.model_trainer.set_model_params(client.model_state_dict)
        stats = self._aided_data_test(round_idx,client)
        client.aid_loss = self.global_loss - stats["aided_test_loss"]

    def train(self):
        #获取全局模型
        w_global = self.model_trainer.get_model_params()
        for values in w_global.values():
                self.model_dim += values.numel()
        # last_model_dict = copy.deepcopy(w_global)
        rd_test_matric_record = [[],[]]
        rd_his_acc = {}
        for round_idx in range(self.args.comm_round):
            w_locals = []
            w_locals_fisher = []
            clientdatnunm = []
            rd_acc = []
            logging.info("\n#######################################################Communication round : {}".format(round_idx))
            clients = []
            # weight_of_w = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            print("self.args.ci:", self.args.ci)
            if self.args.ci == 1:
                if self.args.defend_type == "MAB_FL":  # client_num_per_round是MAB_FL参与的用户总数
                    client_indexes = self._client_sampling(0, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)#random_sample      先随机采样，在用汤普森法进一步选择
                    client_indexes = np.array(client_indexes)
                    attackers_indexes = []
                    for i, index in enumerate(client_indexes):                        #此处攻击者顺序选取
                        if i < self.args.attacker_num:
                            attackers_indexes.append(index)
                    client_indexes = self._Thomson_sampling(round_idx,client_indexes)
                    # 随机选取攻击者
                    # np.random.seed(2)  # make sure for each comparison, we are selecting the same attackers for all round
                    # attackers_indexes = np.random.choice(client_indexes, self.args.attacker_num, replace=False)
                    logging.info("all_attackers_indexes = " + str(attackers_indexes))
                    sellected_attackers_indexes = []
                    for idx in client_indexes:
                        if idx in attackers_indexes:
                            sellected_attackers_indexes.append(idx)
                    logging.info("sellected_attackers_indexes = " + str(sellected_attackers_indexes))
                    #     if round_idx < 10:
                    #         client_indexes = [client_index for client_index in range(self.args.client_num_per_round)]
                    #     else:
                    #         client_indexes = self._Thomson_sampling()
                    #         #攻击者，为id最小的那些客户
                    #     client_indexes = np.array(client_indexes)
                    #     attackers_indexes = client_indexes[np.where(client_indexes < self.args.attacker_num)[0]]
                    # else:
                    #     client_indexes = [client_index for client_index in range(self.args.client_num_per_round)]
                    #     client_indexes = np.array(client_indexes)
                    #     attackers_indexes = client_indexes[np.where(client_indexes < self.args.attacker_num)[0]]
                else:
                    client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                           self.args.client_num_per_round)  # random_sample
                    # 随机选取攻击者
                    attackers_indexes = []
                    for i, index in enumerate(client_indexes):
                        if i < self.args.attacker_num:
                            attackers_indexes.append(index)
                    logging.info("all_attackers_indexes = " + str(attackers_indexes))
                logging.info("sample_client_indexes = " + str(client_indexes)+"num"+str(len(client_indexes)))
                # print("client_list",[c.clientid_x for c in self.client_list])

            else:
                if self.args.sample_method == "cross_silo":
                    client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                           self.args.client_num_per_round)  # random_sample
                    # 随机选取攻击者
                    np.random.seed(
                        2)  # make sure for each comparison, we are selecting the same attackers for all round
                    attackers_indexes = np.random.choice(client_indexes, self.args.attacker_num, replace=False)

                    # round_sample_num = sum([self.train_data_local_num_dict[client_idx] for client_idx in client_indexes])
                    # logging.info("all_client_list_index = " + str([client_indexes[i] for i in range(len(client_list))]))
                    logging.info("sample_client_indexes = " + str(client_indexes))
                    # print("client_list",[c.client_idx for c in self.client_list])

                if self.args.sample_method == "cross_device":
                    metric = {
                        "attacker_num": 0}
                    attackers_indexes = []
                    client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                           self.args.client_num_per_round)  # random_sample
                    for index in client_indexes:
                        if index < self.args.attacker_num:
                            attackers_indexes.append(index)
                            metric["attacker_num"] += 1
                    logging.info("sample_client_indexes = " + str(client_indexes) + "num" + str(len(client_indexes)))
                    logging.info("attackers_indexes = " + str(attackers_indexes))
                    wandb.log({"sample_attacker_num": metric["attacker_num"], "round": round_idx})


            method = ["fedbt", "heterofl"]
            if self.args.defend_type in method:
                client = self.client_list[0]
                client.update_local_dataset(0, 0, self.train_data_local_dict[0],
                                            self.test_data_local_dict[0],
                                            self.train_data_local_num_dict[0], self.train_labels_type_dic[0],self.global_test_aided_data)
                self.model_trainer.set_model_params(w_global)
                stats = self._aided_data_test(round_idx,client)
                self.global_loss = stats["aided_test_loss"]
                print("global_aided:loss",self.global_loss)

            # print("test_flag:",[c.test_num for c in self.client_list])
            for idx, client in enumerate(self.client_list):#client_list 为所有客户的instance
                # update dataset。为什么还要update dataset？因为随机采样后，客户的instance是前面几个客户，而索引是不一样的，所以必须更新数据集以使得两者对应


                if self.args.defend_type == "MAB_FL":
                    if idx >= len(client_indexes):break
                    client_idx = client_indexes[idx]
                    client.model_dim = self.model_dim
                    # print("client_idex{}_test_num{}".format(client.client_idx,client.test_num))

                    client.update_client_MAB_FL_setting(round_idx,self.client_selected_record_dict[client_idx],
                                                        self.client_succ[client_idx],self.client_fail[client_idx])
                else:
                    client_idx = client_indexes[idx]
                    client.model_dim = self.model_dim

                if client_idx in attackers_indexes:
                    client.update_local_dataset(client_idx,True,self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx],self.train_labels_type_dic[client_idx],self.auxiliary_data_dict[client_idx])
                else:
                    client.update_local_dataset(client_idx,False,self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx],self.train_labels_type_dic[client_idx],self.auxiliary_data_dict[client_idx])
                # train on new dataset

                _,updata = client.train(round_idx,copy.deepcopy(w_global))
                # updata = self.get_local_grad(self.args,self.device,copy.deepcopy(w),copy.deepcopy(w_global))
                client.class_num =self.class_num - 1
                client.grad = updata

                for key in w_global.keys():
                    # if RAG.is_weight_param(key):
                    client.model[key] = w_global[key].to(self.device) - client.grad[key].to(self.device)

                # client.model_state_dict = copy.deepcopy(w)
                #client.vec_grad_one = 0
                client.vec_grad_one = self.danweihua(client.grad,self.device)
                print(client.vec_grad_one)
                #client.vec_grad_no_meanvar = 0
                client.vec_grad_no_meanvar = RAG.vectorize_weight(client.grad,self.device)
                client.grad_norm = torch.norm(RAG.vectorize_state_dict(client.grad,self.device),2)
                client.grad_norm_wo_mean_var = torch.norm(RAG.vectorize_state_dict(client.grad,self.device),2)
                # print("client_norm",client.vec_grad_one[:5])
                method = ["fedbt"]
                if self.args.defend_type in method:
                    self.test_model(round_idx,client,copy.deepcopy(w_global))
                clients.append(client)


                # # #///本地测试
                # if round_idx % self.args.frequency_of_the_test == 0:#设置记录准确率的频率
                #     if self.args.dataset.startswith("stackoverflow"):
                #         self._local_test_on_validation_set(round_idx)
                #     else:
                #         self._local_test_on_client(round_idx,client)



            Aggregator_method_obj = RobustAggregator(self.args)  # 设置鲁邦聚合类的资源
            #model_posioning_attack
            rd_his_acc[round_idx] = rd_acc
            loss = []
            for c in clients:
                loss.append([c.byzantine,c.aid_loss])
            # print("round_inx{}_noatk_loss:{}".format(round_idx,loss))
            if self.args.attack_type == "model_attack":
                atk = Model_Attacker(round_idx,self.args,self.device,self.model_trainer,Aggregator_method_obj,self._aided_data_test,copy.deepcopy(w_global))
                atk.global_loss = self.global_loss
                atk.test_global = self.test_global
                atk.test_global_aided = self.global_test_aided_data

                if self.args.attacker_knowlege in self.atk_knowlege:
                    print("dnc_attack")
                    clients = atk.dnc_attack(clients)
                if self.args.attacker_knowlege == "a_little":
                    clients = atk.a_little_attack(clients)
                if self.args.attacker_knowlege == "sign_flipping":
                    print("sign_flipping")
                    atk.sign_flipping(clients)
                if self.args.attacker_knowlege == "random_grad":
                    print("random_grad")
                    atk.random_grad(clients)
            if self.args.attack_type == "PESS_attack":

                atk = Model_Attacker(round_idx,self.args,self.device,self.model_trainer,Aggregator_method_obj,self._aided_data_test,copy.deepcopy(w_global))
                clients = atk.PESS_attack(atk,clients)

            # test = []
            # for i,cc in enumerate(clients):
            #     test.append([cc.byzantine,cc.client_idx,cc.succ,cc.fail])
            #     print("current_test_BYT_ID",test[i])
            if self.args.defend_type == 'fedavg':
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))
            if self.args.defend_type == 'norm_diff_clipping':
                w_locals_norm_diff_clipping = []
                for (client_sample_number,w) in w_locals:
                    # print(w.state_dict)
                    w_clip = Aggregator_method_obj.norm_diff_clipping(w,w_global)
                    w_locals_norm_diff_clipping.append((client_sample_number, w_clip))
                # update global weights
                w_global = self._aggregate(w_locals_norm_diff_clipping)  # 按模型加权聚合
            if self.args.defend_type == 'add_week_noise':
                w_locals_norm_diff_clipping_noise = []
                for (client_sample_number,w) in w_locals:
                    # print(w.state_dict)
                    w_clip = Aggregator_method_obj.norm_diff_clipping(w,w_global)
                    # for (k,v) in w_clip.items():
                    w_clip_noise = Aggregator_method_obj.add_noise(w_clip,self.device)
                    w_locals_norm_diff_clipping_noise.append((client_sample_number, w_clip_noise))
                    w_global = self._aggregate(w_locals_norm_diff_clipping_noise)  # 按模型加权聚合

            if self.args.defend_type == 'Zeno':
                # w_global = Aggregator_method_obj.Zeno(round_idx,g_locals,copy.deepcopy(w_global),self.test_global,self.args,self.device,self.model_trainer)
                global_grad = Aggregator_method_obj.Zeno(round_idx, clients, copy.deepcopy(w_global), self.test_global,
                                                  self.args, self.device, self.model_trainer,False)
                w_global = self._grad_desc(global_grad, w_global)
            if self.args.defend_type == 'fltrust':
                # w_global = Aggregator_method_obj.Zeno(round_idx,g_locals,copy.deepcopy(w_global),self.test_global,self.args,self.device,self.model_trainer)
                global_grad = Aggregator_method_obj.fltrust(round_idx, clients, copy.deepcopy(w_global), self.test_global,
                                                  self.args, self.device, self.model_trainer)
                w_global = self._grad_desc(global_grad, w_global)
            if self.args.defend_type == 'DiverseFL':
                w_global = Aggregator_method_obj.DiverseFL(round_idx,g_locals, copy.deepcopy(w_global), self.args, self.device, self.model_trainer)
            if self.args.defend_type == 'Fisher':
                if round_idx < 1 :
                    loc = []
                    round_diff = {}
                w_locals_fisher = Aggregator_method_obj.Fisher(round_idx,loc,round_diff,w_global,w_locals_fisher,self.args,self.device,self.model_trainer)
                w_global = self._aggregate(w_locals_fisher)  # 按模型加权聚合
            if self.args.defend_type == 'MKrum':
                global_grad = Aggregator_method_obj.MKrum(round_idx,False,clients,self.args,self.device)
                w_global = self._grad_desc(global_grad, w_global)
                # w_global = self._aggregate(w_locals_fisher)  # 按模型加权聚合
            if self.args.defend_type == "test":
                if round_idx < 1 :
                    round_diff = {}
                    rd_drt = []
                    rd_loc = np.random.choice(range(RAG.vectorize_weight(w_global).numel()), 5, replace=False)
                # print("rd:{}_g_w{}".format(round_idx,RAG.vectorize_weight(w_global)[rd_loc]))
                w_locals = Aggregator_method_obj.test(round_idx,rd_loc,rd_drt,round_diff,w_global,w_locals_fisher,self.args,self.device,self.model_trainer,original_model)
                # print("rd:{}_diff{}".format(round_idx,round_diff[round_idx]))
                print("rd_drt",rd_drt)
                w_global = self._aggregate(w_locals)  # 按模型加权聚合
            if self.args.defend_type == 'median':
                global_grad = Aggregator_method_obj.median(round_idx,copy.deepcopy(w_global), clients,
                                                               self.args, self.device, self.model_trainer)
                w_global = self._grad_desc(global_grad, w_global)
            if self.args.defend_type == 'resample':
                rasample_mean_grad,resample_norm = Aggregator_method_obj.resample(round_idx, clients, self.args,self.device)
                for c_idx,c in enumerate(clients):
                    mean_grad = rasample_mean_grad[c_idx].cpu().numpy().tolist()
                    grad_dict = RAG.recover_to_dict(mean_grad, w_global, self.device)
                    c.class_num = self.class_num - 1
                    c.grad = grad_dict
                    # client.model_state_dict = copy.deepcopy(w)
                    c.vec_grad_one = rasample_mean_grad[c_idx]
                    # client.vec_grad_no_meanvar = 0
                    c.vec_grad_no_meanvar = RAG.vectorize_weight(c.grad, self.device)
                    c.model_dim = c.vec_grad_one.numel()
                    # print("client.model_dim", c.model_dim)
                    c.grad_norm = resample_norm[c_idx]
                    # print("client_norm",client.vec_grad_one[:5])
                global_grad = Aggregator_method_obj.MKrum(round_idx,False,clients,self.args,self.device)
                w_global = self._grad_desc(global_grad, w_global) # 按模型加权聚合
            if self.args.defend_type == 'faba':
                global_grad = Aggregator_method_obj.faba(round_idx, clients,
                                                        self.args, self.device)
                w_global = self._grad_desc(global_grad, w_global)
            if self.args.defend_type == 'expsmoo':
                alpha = 0.8
                # if round_idx % 10 == 0 or round_idx == 0:   #每10轮后重置预测模型，因为指数平滑适用于短期预测测
                if round_idx == 0:
                    sample_num = 20
                    data_x, data_y = RAG.undo_batch_dataset(self.test_global)
                    zeno_val_data = RAG.get_zeno_val_data(sample_num, data_x, data_y)
                    for i in range(self.args.epochs*self.train_data_local_num_dict[0]//self.args.batch_size):
                        s1 = self.model_trainer.expsmoo_train(round_idx,zeno_val_data,self.device, self.args,copy.deepcopy(w_global))
                    # s1 = self.model_trainer.get_model_params()
                    s2 = s1
                global_grad = Aggregator_method_obj.expsmoo(round_idx,clients,alpha,s1,s2,self.args, self.device, self.model_trainer)
                w_global = self._grad_desc(global_grad, w_global)
                for na, para in w_global.items():
                    s1[na] = alpha * w_global[na] + (1 - alpha) * s1[na].to(self.device)
                    s2[na] = alpha * s1[na].to(self.device) + (1 - alpha) * s2[na].to(self.device)
            if self.args.defend_type == 'fedbt':
                global_grad = Aggregator_method_obj.fedbt(round_idx,clients,False,copy.deepcopy(w_global),self.args,self.device,self.model_trainer)
                w_global = self._grad_desc(global_grad, copy.deepcopy(w_global))
                # print("2")
                # if Aggregator_method_obj.test_FLAG:
                #     std_wrt_rate = np.array([self.args.partition_alpha,Aggregator_method_obj.clients_cos_std,Aggregator_method_obj.group_cos_std])
                #     cos_sum_wrt_rate = np.array([self.args.partition_alpha, Aggregator_method_obj.clients_cos_sum/1225,
                #                              Aggregator_method_obj.group_cos_sum/Aggregator_method_obj.group_fenzi])
                #     print("cos_sum_wrt_rate",cos_sum_wrt_rate)
                #     with open(r"C:\Alvin\FedML-master\fedml_experiments\standalone\fedavg\atknum_{}_cos_sum_wrt_rate.txt".format(self.args.attacker_num), 'a+') as f:
                #         f.write(str(cos_sum_wrt_rate) + '\n')
            if self.args.defend_type == "MAB_FL":
                fianal_sellected_clients,global_grad = Aggregator_method_obj.MAB_FL(round_idx,clients,False,copy.deepcopy(w_global),self.args,self.device,self.model_trainer)
                w_global = self._grad_desc(global_grad, copy.deepcopy(w_global))

                for c in clients:
                    if round_idx>=1:
                        # print("record",self.client_selected_record_dict[c.client_idx])
                        self.client_selected_record_dict[c.client_idx] = round_idx
                    if c in fianal_sellected_clients:
                        self.client_succ[c.client_idx] += 1
                    else:self.client_fail[c.client_idx] += 1

            if self.args.defend_type == "dnc":
                global_grad = Aggregator_method_obj.dnc(round_idx, clients, False,
                                                        self.args, self.device)
                w_global = self._grad_desc(global_grad, copy.deepcopy(w_global))
            # if self.args.defend_type == 'Resample':
            # if self.args.defend_type == 'RFA':

            # last_model_dict = copy.deepcopy(w_global)
            self.model_trainer.set_model_params(w_global)
            _ = self._global_test(round_idx)
        # if self.args.comm_round == round_idx+1:
        if os.path.exists('./finalmodel/'+str(self.args.model)) !=True:#attacker_knowlege = esorics
            torch.save(w_global,"./finalmodel/"+str(self.args.model))


            # rd_test_matric_record[0].append(matric_state["global_Acc"])
            # rd_test_matric_record[1].append(matric_state["global_Loss"])
            # record_name  = ["global_Acc","global_Loss"]
            # if os.path.exists('./save_data/'+str(self.args.attacker_knowlege)) !=True:#attacker_knowlege = esorics
            #     os.makedirs('./save_data/'+str(self.args.attacker_knowlege)) #+'/'+str(self.args.defend_type)
            # if os.path.exists('./save_data/'+str(self.args.attacker_knowlege)+'/'+str(self.args.attack_type)) !=True:
            #     os.makedirs('./save_data/' + str(self.args.attacker_knowlege) + '/' + str(self.args.attack_type))
            #     # os.makedirs('./save_data/' + str(self.args.attacker_knowlege) + '/atk' + str(self.args.attacker_num))
            # if os.path.exists('./save_data/' + str(self.args.attacker_knowlege)+'/'+str(self.args.attack_type) + '/atk' + str(self.args.attacker_num)) !=True:
            #     os.makedirs('./save_data/' + str(self.args.attacker_knowlege)+'/'+str(self.args.attack_type) + '/atk' + str(self.args.attacker_num))
            #     # os.makedirs('./save_data/' + str(self.args.attacker_knowlege) + '/atk' + str(self.args.attacker_num))
            # sav_path = './save_data/' + str(self.args.attacker_knowlege)+'/'+str(self.args.attack_type) + '/atk' + str(self.args.attacker_num) +'/'
            # pd.DataFrame(index = record_name,data=rd_test_matric_record).to_csv(sav_path + str(self.args.defend_type)+'.csv')

    def _client_sampling(self,round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)# 每轮选不同用户
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    def _Thomson_sampling(self,round_idx,clients_index):
        client_indexes = []  #被选中的用户的列表
        if round_idx < 10:
            return clients_index
        else:
            pro = {}
            pro[round_idx] = []
            for i in clients_index:
                p = np.random.beta(self.client_succ[i], self.client_fail[i])
                pro[round_idx].append(p)
                if p >= 0.9:
                    client_indexes.append(i)
                if p > 0.2 and p <= 0.9 and np.random.random() < p:
                    client_indexes.append(i)
            if len(client_indexes) == 0:  # 如果本轮采样结果为空，则所有用户均被选中
                client_indexes = clients_index
            logging.info("each_round_pro:{}".format(pro[round_idx]))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset
    def _grad_desc(self, global_grad,current_model):#加权聚合
        for key, param in current_model.items():

            # if RAG.is_weight_param(key):
                current_model[key] = param.to(self.device) - global_grad[key]

            # else:
            #     current_model[key] = global_grad[key]
        return current_model

    def _aggregate(self, w_locals):#加权聚合
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                    # print("_aggregate1")
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_client(self, round_idx,client):
        logging.info("\n")
        logging.info("local_test_on_client_{}_at_round : {}".format(client.client_idx,round_idx))

        """
        Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
        the training client number is larger than the testing client number
        """
        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        ### train data
        train_local_metrics = client.local_test(1) #False使用训练集
        train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
        train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
        train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

        ### test data
        test_local_metrics = client.local_test(2)#Ture使用验证集
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))



        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        ## test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])



        # stats = {'training_acc': train_acc, 'training_loss': train_loss,'client_idx': client.client_idx}
        logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Train/Acc": train_acc, "Train/Loss": train_loss})
        stats = {'test_acc': test_acc, 'test_loss': test_loss,'client_idx': client.client_idx}
        logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Test/Acc": test_acc, "Test/Loss": test_loss})
        # logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Train/Acc": train_acc, "Train/Loss": train_loss})

        # return pubulic_stats

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients_at_round : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]
        client.model_trainer.set_model_params(client.local_model)
        # print("client.local_model",client.local_model)
        # print("client.model_trainer.set_model_params(client.local_model)",self.model_trainer.get_model_params())
        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0,client.byzantine, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx],self.train_labels_type_dic[client_idx])
            # train data
            train_local_metrics = client.local_test(1)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(2)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            if client_idx == self.args.client_num_per_round :
                break
        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)


    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, client.byzantine,None, self.val_global, None,None)
        # test data
        test_metrics = client.local_test(2)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)
    def _backdoor_task_test(self,round_idx):
        logging.info("\n")
        logging.info("global_test_on_validation_set_at_round : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        sever = self.client_list[0]  # 代用客户1来测试全局模型准确率
        sever.update_local_dataset(0, sever.byzantine, None, self.targetted_task_test_loader, None, None, None)  # 设置全局验证数据
        # test data
        # print("set_gl_model",sever.model_trainer.get_model_params())
        test_global_metrics = sever.global_test()  # 使用全局模型在验证数据集上测试
        test_metrics['num_samples'].append(copy.deepcopy(test_global_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_global_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_global_metrics['test_loss']))

        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        stats = {'backdoor_Acc': test_acc, 'backdoor_Loss': test_loss}
        wandb.log({"backdoor_Acc": test_acc, "round": round_idx})
        wandb.log({"backdoor_Loss": test_loss, "round": round_idx})
        logging.info(stats)
        return stats
    def _global_test(self,round_idx):
        logging.info("\n")
        logging.info("global_test_on_validation_set_at_round : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        sever = self.client_list[0] #代用客户1来测试全局模型准确率
        sever.update_local_dataset(0,sever.byzantine,None,self.test_global, None,None,None)#设置全局验证数据
        # test data
        # print("set_gl_model",sever.model_trainer.get_model_params())
        test_global_metrics = sever.global_test()#使用全局模型在验证数据集上测试
        test_metrics['num_samples'].append(copy.deepcopy(test_global_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_global_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_global_metrics['test_loss']))

        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        stats = {'global_Acc': test_acc, 'global_Loss': test_loss}
        wandb.log({"global_Acc": test_acc, "round": round_idx})
        wandb.log({"global_Loss": test_loss, "round": round_idx})
        logging.info(stats)
        return stats
    def _aided_data_test(self,round_idx,client):

        if self.args.attack_type != "model_attack":
            logging.info("\n")
            logging.info("****Aided_data_test_on_cliet_{}_at_round : {}".format(client.client_idx,round_idx))

        """
        Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
        the training client number is larger than the testing client number
        """

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        # test data
        test_local_metrics = client.local_test(3)#使用辅助验证集
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        # logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Train/Acc": train_acc, "Train/Loss": train_loss})
        stats = {'aided_test_acc': test_acc, 'aided_test_loss': test_loss,'client_idx': client.client_idx}
        if self.args.attack_type != "model_attack":
            logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Aided_Test/Acc": test_acc, "Aided_Test/Loss": test_loss})
            logging.info(stats)
        # else:
        #     print("searching...")
        return stats