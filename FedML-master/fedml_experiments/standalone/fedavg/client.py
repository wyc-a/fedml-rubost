import logging
import torch
import collections
import copy
from torch import nn

def is_weight_param(k):
    return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number,train_labels_type, args, device,
                 model_trainer,aided_data):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.aided_test_data = aided_data
        self.labels = train_labels_type
        # logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.local_model = None
        self.guding_model = None
        self.acc_grad = None
        self.grad = None
        self.grad_norm = 0
        self.grad_norm_wo_mean_var = 0
        self.vec_grad_one = None
        self.model = collections.OrderedDict()
        self.model_state_dict = None
        self.model_norm = None
        self.model_dim =None
        self.vec_model_one = None
        self.vec_grad_no_meanvar =None
        self.loss = nn.CrossEntropyLoss()
        self.byzantine = False
        self.weight = 1/self.args.client_num_per_round
        self.min_norm = None
        self.aid_loss = 0
        self.class_num =0
        self.test = 0
        self.momentum = None
        self.succ = 1
        self.fail = 1
        self.seleted_epoch = 0
        self.miu = 0.1
        self.test_num = 0
        # self.set_guding_model = False
    def update_client_MAB_FL_setting(self,round_idx,client_selected_record,client_succ,client_fail):

        self.succ = client_succ
        self.fail = client_fail
        self.seleted_epoch = client_selected_record
    def update_local_dataset(self, client_idx, client_idx_state,local_training_data, local_test_data, local_sample_number,lables,aided_test_data):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.labels = lables
        self.byzantine = client_idx_state
        # self.acc_grad
        self.aided_test_data = aided_test_data
    def accmulate_grad(self,curr_grad):
        if self.acc_grad == None:
            self.acc_grad = curr_grad
        else:
            for key,param in curr_grad.items():
                self.acc_grad[key] += param

    def get_sample_number(self):
        return self.local_sample_number

    def train(self,round_idx, w_global):#w_global为字典
        logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        if self.byzantine == False:
            grads = self.model_trainer.train(round_idx,self.local_training_data, self.device, self.args,w_global)
            weights = self.model_trainer.get_model_params()
            # weights = 1
        else:#攻击者执行攻击
            # print("1")
            self.attack_based_on_data() #None表示无基于参数的攻击
            # print("3")
            grads = self.model_trainer.train(round_idx,self.local_training_data,self.device,self.args,w_global)
            weights = self.model_trainer.get_model_params()
            grads,weights, = self.attack_based_on_param(grads,weights)
        return weights,grads

    def local_test(self, b_use_test_dataset):

        if b_use_test_dataset == 1:
            test_data = self.local_training_data
            # print("client_model_ture",self.model_trainer.get_model_params())
        if b_use_test_dataset == 2:
            test_data = self.local_test_data
            # print("client_model_False",self.model_trainer.get_model_params())
        if b_use_test_dataset == 3:
            test_data = self.aided_test_data
            # print("client_model_False",self.model_trainer.get_model_params().items())
        metrics = self.model_trainer.test(test_data, self.device, self.args,self.guding_model,self.local_model)
        return metrics

    def global_test(self):
        test_data = self.local_test_data
        # print("global_model_T", self.model_trainer.get_model_params())
        metrics = self.model_trainer.global_test(test_data, self.device, self.args)
        return metrics
    def attack_based_on_data(self):
        # model = []
        # print("2.1",self.args.attack_type)
        if self.args.attack_type == "label_flipping":

            flipped_data = []
            model = ["mobilenet","lr"]
            if self.args.model in model:
                # print("2.2")
                logging.info("label_flipping...")

                for (x,label) in self.local_training_data:
                    lbelow = label[torch.where(label < 10)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 9
                    label[torch.where(label < 10)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if self.args.model == "cnn" and self.args.dataset == "mnist":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    lbelow = label[torch.where(label < 10)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 9
                    label[torch.where(label < 10)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if self.args.dataset == "cifar10":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    lbelow = label[torch.where(label < 9)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 8
                    label[torch.where(label < 9)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if  self.args.model == "cnn" and self.args.dataset == "femnist":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    # print("x",x.size())
                    lbelow = label[torch.where(label<62)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 61
                    label[torch.where(label<62)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if  self.args.model == "rnn" and self.args.dataset == "shakespeare":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    # print("pre_label:",label[:20])
                    loc_below = torch.where(label<81)[0]
                    loc_above = torch.where(label>81)[0]
                    lbelow = label[loc_below]
                    l_1 = torch.ones(size=lbelow.size(), dtype=torch.int) * 90
                    labove = label[loc_above]
                    l_2 = torch.ones(size=labove.size(), dtype=torch.int) * 90
                    label[loc_below] = l_1 - lbelow
                    label[loc_above] = l_2 - labove
                    print("new_label:", label[:20])
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
        # if self.args.attack_type == "bitflip_attack":
        #     for (x,label) in self.local_training_data:
        #         x[:] = -x[0]

    def attack_based_on_param(self, grads, weight):

        if self.args.attack_type == "gaussian_attack":
            print("self.args.attack_type", self.args.attack_type)
            for (name,v) in weight.items():
                # if is_weight_param(name):
                print("name",name)
                gaussian_noise = torch.normal(mean=0,std=0.5,size=v.size()).to(self.device)
                # grads[name] = grads[name].to(self.device) + gaussian_noise
                weight[name] = weight[name].to(self.device) + gaussian_noise

        if self.args.attack_type == "sign_flipping":
            print("self.args.attack_type", self.args.attack_type)
            for (name,_) in weight.items():
                grads[name] = -20*grads[name] #2.5
                weight[name] = -20*weight[name]
        # if self.args.attack_type == "random_grad":
        #     print("self.args.attack_type", self.args.attacker_knowlege)
        #     for (name, parm) in grads.items():
        #         grads[name] = torch.normal(mean=0, std=2, size=parm.size()).div(1e20)
        #         weight[name] = torch.normal(mean=0, std=2, size=parm.size()).div(1e20)


        # if self.args.attack_type == "param_flipping":
        #     print("self.args.attack_type", self.args.attack_type)
        #     grads = 0
        #     weight = 0
        return grads,weight
    # def attack_based_parameter(self,weights):
    #     return weights





