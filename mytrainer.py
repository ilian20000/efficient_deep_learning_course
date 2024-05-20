import torch
import time
import sys

import mydataloader

import matplotlib.pyplot as plt

class NetTrainer():

    def __init__(self, net, args):
        self.debug = args.debug
        self.net = net
        self.half = False

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=10, momentum=0.9)
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lambda_lr = lambda epoch: 0.98**epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)

        self.net.to(self.device)
        self.augment_type = 2
        self.update_datasets()

        self.epoch = 0
        self.nepochs = 0
        self.epochtime = -1
        self.benchstats = {"epoch":[], 
                           "lr":[],
                           "train accuracy":[], 
                           "test accuracy":[],
                           "train loss":[]}
        print("Target device :", self.device)

    def __str__(self) -> str:
        str_desc = ""
        return str_desc
        
    def update_datasets(self):
        self.trainset = mydataloader.load_trainset(half=self.half, debug=self.debug, augment=self.augment_type)
        self.testset =  mydataloader.load_testset(half=self.half, debug=self.debug)
        # for i, data in enumerate(self.trainset):
        #     imgs, labels = data
        #     if i==0:
        #         fig, ax = plt.subplots(4,4)
        #         for l in range(4):
        #             for m in range(4):
        #                 img = imgs[l + 4*m].T
        #                 ax[l, m].imshow(img)
        # plt.show()
 

    def save(self, filepath):
        state = {
        'net': self.net.state_dict(),
        'benchstats': self.benchstats,
        'epochtime': self.epochtime,
        'epoch' : self.epoch
    }
        torch.save(state, filepath)
    

    def load(self, filepath):
        loaded_cpt = torch.load(filepath)
        self.net.load_state_dict(loaded_cpt['net'])
        self.benchstats = loaded_cpt['benchstats']
        self.epoch = loaded_cpt['epoch']
        self.net.to(self.device)


    def to_half(self):
        self.half = True
        self.net.half()
        self.update_datasets()

    def singleloop(self):
        running_loss = 0.0
        t_0 = time.time()
        self.net.train()
        for i, data in enumerate(self.trainset):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i%100 == 0:
                print(f'Epoch n°{self.epoch} - lr={self.scheduler.get_lr()[0]} - Dataset progress : {i}/{len(self.trainset)}', end='\r', flush=True)
        self.scheduler.step()
        mean_loss = running_loss/len(self.trainset)
        loop_time = time.time() - t_0
        loop_stats = (mean_loss, loop_time)
        return loop_stats
    

    def trainloop(self, n_eval=10):
        print("Started trainloop")
        t_start = time.time()
        while self.epoch < self.nepochs:
            self.epoch += 1

            t_0 = time.time()
            mean_loss, loop_time = self.singleloop()
            t_1 = time.time() - t_0

            if self.epoch == 1:
                t_estimate = int(t_1*self.nepochs)
                t_hour = t_estimate//3600
                t_min = t_estimate//60 - t_hour*60
                t_sec = t_estimate - t_min*60  - t_hour*3600
                sys.stdout.write("\033[K")
                print(f"Time estimated : {t_hour}h {t_min}min {t_sec}s")
                self.epochtime = t_1

            if (self.epoch)%n_eval == 0:
                train_rate, test_rate = self.benchmark()
                self.benchstats["train loss"].append(mean_loss)
                self.benchstats["epoch"].append(self.epoch)
                self.benchstats["lr"].append(self.scheduler.get_lr()[0])
                self.benchstats["train accuracy"].append(train_rate)
                self.benchstats["test accuracy"].append(test_rate)
                self.save(f"saves/resnetcifarSCHEDULEepoch")

            # print(f'epoch n°{self.epoch} ended in {t_1} seconds')
        pass


    def benchmark(self):
        correct_test = 0
        total_test = 0
        self.net.eval()

        for data in self.testset:
            imgs, labels = data
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            output = self.net(imgs)
            _, predict = torch.max(output, 1)

            total_test += labels.size(0)
            correct_test += (labels == predict).sum().cpu().numpy()

        correct_train = 0
        total_train = 0

        for data in self.trainset:
            imgs, labels = data
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            output = self.net(imgs)
            _, predict = torch.max(output, 1)

            total_train += labels.size(0)
            correct_train += (labels == predict).sum().cpu().numpy()        

        test_rate = correct_test/total_test
        train_rate = correct_train/total_train
        sys.stdout.write("\033[K")
        print(f"Epoch n°{self.epoch} - lr={self.scheduler.get_lr()[0]} Test Acc={test_rate*100}% Train Acc={train_rate*100}%")
        return train_rate, test_rate