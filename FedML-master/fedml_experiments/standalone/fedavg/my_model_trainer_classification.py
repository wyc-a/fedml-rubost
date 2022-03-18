import torch
import collections
import numpy as np
import copy
from torch import nn
from fedml_api.standalone.fednova import fednova
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer

# class MyHingeLoss(torch.nn.Module):
#     # 不要忘记继承Module
#     def __init__(self):
#         super(MyHingeLoss, self).__init__()
#
#     def forward(self, output, target):
#         """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
#         """
#         hinge_loss = 1 - torch.mul(output, target)
#         hinge_loss[hinge_loss < 0] = 0
#         # 不要忘记返回scalar
#         return torch.mean(hinge_loss)

class MyModelTrainer(ModelTrainer):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def vectorize_weight(self,state_dict,device):
        weight_list = []
        for (k, v) in state_dict.items():
            if self.is_weight_param(k):  #
                weight_list.extend(v.view(1, -1).to(device))
        return torch.cat(weight_list)  # torch.cat的用法必须注意
    def vectorize_model(self,model):
        weight_list = []
        for k, v in model.named_parameters():
            if self.is_weight_param(k):  #
                weight_list.extend(v.view(1, -1))
        return torch.cat(weight_list)  # torch.cat的用法必须注意
    def is_weight_param(self,k):
        return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)

    def compute_loss(self,model_state_dict,data_x,data_y,device):
        self.set_model_params(model_state_dict)
        model = self.model
        model.to(device)
        model.eval()
        loss = nn.CrossEntropyLoss()
        criterion = loss.to(device)
        x, labels = data_x.to(device), data_y.to(device)
        # model.zero_grad()
        log_probs = model(x)

        return criterion(log_probs, labels).item()
    def get_server_model(self,args,model_state_dict,sample_data,device):
        # print(model_state_dict)
        self.set_model_params(model_state_dict)
        init_params = copy.deepcopy(model_state_dict)
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        # train and updates
        criterion = loss.to(device)
        epoch_grad = collections.OrderedDict()
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        for epoch in range(args.epochs):  # 本地训练的轮次。
            for batch_idx, (x, labels) in enumerate(sample_data):
                # print("batch:",batch_idx,x.size(),labels)
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
        return self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)

    # L2 正则化
    def L2Losss(self,model, alpha):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, parma in model.named_parameters():
            if self.is_weight_param(name):
                l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(parma, 2)))
        return l2_loss
    def get_local_grad(self,args, device,cur_params, init_params):
        update_dict = collections.OrderedDict()
        for k in cur_params.keys():
            # if self.is_weight_param(k):
            update_dict[k] = init_params[k].to(device) - cur_params[k].to(device)
            # else:update_dict[k] = cur_params[k].to(device)
        return update_dict
    # 约束
    def add_constraint_loss(self,model, last_model_dict,alpha,device):
        l2_loss = torch.tensor(0.0, requires_grad=True)

        for name, parma in model.named_parameters():

            if self.is_weight_param(name):
                # print("now_gw", parma)
                # print("last_gw", last_model_dict[name])
                deta = torch.abs(parma - torch.tensor(last_model_dict[name]).to(device))
                max = torch.max(parma)
                min = torch.min(parma)
                # print("lay:{}_max{}_min{}".format(name,max,min))
                c = torch.ones_like(deta) * 0.01
                mk1 = deta.le(-0.005)
                mk2 = deta.ge(0.005)
                para_to_min = torch.cat([torch.masked_select(parma,mk1),torch.masked_select(parma,mk2)])
                # print("{}_para_to_min{}_{}".format(name,para_to_min.numel(),para_to_min))
                para_to_min = torch.abs(para_to_min)
                #
                # parma = torch.where(deta <c,parma,max)
                # parma = torch.where(deta >-c, parma, min)
                # deta = deta - c
                l2_loss = l2_loss + (0.005 * alpha * torch.sum(para_to_min))

        return l2_loss
    def drtloss(self,model, drt_w_dict,alpha,b,device):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, parma in model.named_parameters():
            if self.is_weight_param(name):
                ls = torch.norm(b*parma - drt_w_dict[name], 1)
                alpha  = 0.0001
                l2_loss = l2_loss + alpha*ls
        return l2_loss
    def kl_loss(self,vec_loc,vec_c_w, vec_g_w,theta,device,epoc_opt_w_num):
        diff = vec_c_w - vec_g_w
        print("diff",diff)
        # dff_max = torch.max(diff)
        # dff_min = torch.min(diff)
        # theta = 1/(dff_max-dff_min)
        # tanh_diff = torch.tanh(theta*diff)
        # print("tanh_diff",tanh_diff)
        dff_sumdiff = theta*diff + torch.ones_like(diff)*theta
        sgmid_wdiff = torch.sigmoid(dff_sumdiff)
        sg_df_chsed = torch.where(sgmid_wdiff < 0.88)[0]
        sgmid_wdiff_ched = torch.index_select(sgmid_wdiff,0,sg_df_chsed)
        print("sgmid_wdiff_ched", sgmid_wdiff_ched)
        sgmid_loc = torch.sigmoid(vec_loc)
        sg_loc_ched = sgmid_loc[:sg_df_chsed.numel()]
        epoc_opt_w_num.append(sg_df_chsed.numel())
        # tanh_loc = torch.tanh(vec_loc)
        # print("tanh_loc", tanh_loc)
        # tanh_wdiff_log = torch.log(torch.tanh(theta * diff))
        # tanh_loc_log = torch.log(torch.tanh(vec_loc))
        # sgmid_wdiff_log = torch.log(torch.sigmoid(theta * diff))
        # sgmid_loc_log = torch.log(torch.sigmoid(vec_loc))
        sgmid_wdiff_log = torch.log(sgmid_wdiff_ched)
        sgmid_loc_log = torch.log(sg_loc_ched)
        return torch.mean(sg_loc_ched * (sgmid_loc_log - sgmid_wdiff_log))

    # def hinge_loss(self,vec_loc,vec_c_w, vec_g_w,theta,device):
    #     diff = vec_c_w - vec_g_w
    #
    #     return torch.mean(sgmid_loc * (sgmid_loc_log - sgmid_wdiff_log))

    def StepB(self,round,args):
        if round<5: b = 4
        elif round <10: b = 1.5
        elif round<15: b = 1.0
        elif round<20:b = 0.25
        elif round<25: b = 0.25
        elif round<30:b = 0.125
        elif round<90:b = 0.100
        else: b = 0.0100
        return b
    def Steptheta(self,round,args):
        if round<5: b = 100
        elif round <10: b = 80
        elif round<15: b = 60
        elif round<20:b = 50
        elif round<25: b = 40
        elif round<30:b = 30
        elif round<90:b = 20
        else: b = 10
        return b
    def check(self,client_model, last_model_dict, sup_drt, sup_step,train_loss,x,labels,device):
        loss = self.compute_loss(client_model,x,labels,device)
        vec_cw = self.vectorize_weight(client_model).to(device)
        vec_lw = self.vectorize_weight(last_model_dict).to(device)
        neg_drtsum =torch.sum(torch.where(torch.sign(vec_cw-vec_lw)<0)[0],dim=0).item()
        step_norm = torch.sum(torch.abs(vec_cw - vec_lw)).item()
        loss_diff = torch.abs(torch.tensor(loss-train_loss))
        print("supdrt:{}_supstep:{}".format(sup_drt,vec_lw.numel()*sup_step))
        print("ls_loss:{}_loss:{}_negdrtsum:{}_stepnorm:{}".format(train_loss,loss,neg_drtsum,step_norm))
        if (step_norm < sup_step)&(neg_drtsum < sup_drt): #应该减少的r
            return True
        else:return False

    def expsmoo_train(self, round_idx, train_data, device, args,last_model_dict):
        model = self.model  # 这个model不是字典，需要get_model_state_dict

        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)
        return updata

    def train(self,round_idx, train_data, device, args,last_model_dict):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        # train and updates
        criterion = loss.to(device)
        init_params = copy.deepcopy(last_model_dict)
        method = ["fedavg","heterofl","Zeno","median","resample","faba","fltrust","fedbt","dnc","fedba","MKrum","expsmoo","MAB_FL"]
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(),lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        epoch_loss = []
        for epoch in range(args.epochs):#本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # print("tt",(batch_idx,labels))
                if epoch == 0 and batch_idx==0:
                    print("x.size",x.size())
                    # print("x:", x)
                    # print("x:", labels)
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                # if args.attack_type == "ESP_attack":#攻击
                #     if args.model == "resnet18":
                #         loss= 0
                #     if args.model == "cnn":loss= 0
                # else:#正常计算分类损失
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                # print("loss:", loss)

                if args.defend_type == 'DiverseFL':
                    loss = loss + self.L2Losss(model,args.i) #L2正则化
                    loss.backward()
                if args.defend_type == 'test':
                    if batch_idx == 0:
                        r = 0.00001
                        step = r / 2
                        # r_succ = 1000
                        sup_drt, sup_step = 2500, 0.02
                        client_model = model.state_dict()
                        while self.check(client_model, last_model_dict, sup_drt, sup_step, loss.item(), x, labels, device) == False:
                            r = r + step / 2
                            for name, parma in client_model.items():
                                if self.is_weight_param(name):
                                    pdiff = client_model[name] - last_model_dict[name].to(device)
                                    pdiff = torch.where(pdiff>0,torch.zeros_like(pdiff),pdiff)
                                    # print("negdff_num",pdiff.numel())
                                    u = torch.where(pdiff < 0, torch.ones_like(pdiff), pdiff)
                                    # print("u_num", u.numel())
                                    client_model[name] = client_model[name] + r * u.to(device)
                            step = 2 * step
                        self.set_model_params(client_model)
                        loss.backward()
                    else:loss.backward()
                if args.defend_type in method:
                    loss.backward()
                optimizer.step() #若出现cuda bug,可能是batchsize 太大导致
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # updata = 1
        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)
        return updata

    def global_test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if args.attack_type == "backdoor":
                    # print(x.size())
                    x = torch.squeeze(x)
                    # print("x.size", x.size())
                    if x.size()[0] == 28:break
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test(self, test_data, device, args,guding_model,client_model):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }
        criterion = nn.CrossEntropyLoss().to(device)
        # label_count = []
        # rd_bt_idx = np.random.choice(range(len(list(enumerate(test_data)))),5,False)
        # print("tdx",rd_bt_idx)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if args.attack_type == "backdoor":
                    # print(x.size())
                    x = torch.squeeze(x)
                    # print("x.size", x.size())
                    if x.size()[0] == 28:break
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                if args.defend_type == 'DiverseFL':
                    loss = loss + self.L2Losss(model,args.i) #L2正则化
                    # loss.backward()
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
