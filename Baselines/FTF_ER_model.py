import torch
from .FTF_ER_utils import *
import dgl
import os

calculators = {'Hodge': Hodge_calculator()}
samplers = {'mix': mix_sampler(plus=False), 'mix_plus': mix_sampler(plus=True)}
class NET(torch.nn.Module):
    """
        FTF-ER baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.
        :param dataset: The entire dataset.
        
        """

    def __init__(self,
                 model,
                 task_manager,
                 args,
                 dataset):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model
        self.sampler = samplers[args.ftfer_args['sampler']]
        self.sampler.beta = args.ftfer_args['beta']
        self.calculator = calculators['Hodge']

        fai_path = os.path.join(args.data_path,args.dataset + ".fai.npy")
        if os.path.exists(fai_path):
            self.fai = np.load(fai_path)
        else:
            self.fai = self.calculator(dataset.graph)
            np.save(fai_path, self.fai)

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = int(args.ftfer_args['budget'])


    def forward(self, features):
        output = self.net(features)
        return output
    
    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
            The method for learning the given tasks under the class-IL setting.

            :param args: Same as the args in __init__().
            :param g: The graph of the current task.
            :param features: Node features of the current task.
            :param labels: Labels of the nodes in the current task.
            :param t: Index of the current task.
            :param train_ids: The indices of the nodes participating in the training.
            :param ids_per_cls: Indices of the nodes in each class.
            :param dataset: The entire dataset.
        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        size_g_train = g.num_nodes()

        aux_train_ids = []

        if t!=self.current_task:
            # if new tasks come, store data from the current task
            self.current_task = t
            self.mix_g = g
            self.mix_train_ids = train_ids
            if t>0:
                last_task_sampled_ids = self.sampler(
                    self.last_task_ids_per_cls_train,
                    self.budget,
                    self.last_task_g,
                    self.net,
                    args,
                    self.last_task_offset1, self.last_task_offset2,
                    self.ce,
                    self.fai
                )
                old_ids = self.last_task_g.ndata['_ID'].cpu()
                self.buffer_node_ids.extend(old_ids[last_task_sampled_ids].tolist())
                aux_g, _, _ = dataset.get_graph(node_ids=self.buffer_node_ids, remove_edges=False)
                old_ids_aux = aux_g.ndata['_ID']
                aux_train_ids = [(old_ids_aux == i).nonzero().squeeze().item() + size_g_train for i in
                                self.buffer_node_ids]
                self.mix_g = dgl.batch([g,aux_g.to(device='cuda:{}'.format(args.gpu))])
                self.mix_train_ids = np.concatenate((train_ids, aux_train_ids))
            self.last_task_g = g
            self.last_task_ids_per_cls_train = ids_per_cls_train
            self.last_task_offset1, self.last_task_offset2 = offset1, offset2

        features, labels = self.mix_g.srcdata['feat'], self.mix_g.dstdata['label'].squeeze()
        if args.cls_balance:
            n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        output, _ = self.net(self.mix_g, features) 
        if args.classifier_increase:
            loss = self.ce(output[self.mix_train_ids, offset1:offset2], labels[self.mix_train_ids], weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset): 
        """
            The method for learning the given tasks under the class-IL setting with mini-batch training.

            :param args: Same as the args in __init__().
            :param g: The graph of the current task.
            :param dataloader: The data loader for mini-batch training
            :param features: Node features of the current task.
            :param labels: Labels of the nodes in the current task.
            :param t: Index of the current task.
            :param train_ids: The indices of the nodes participating in the training.
            :param ids_per_cls: Indices of the nodes in each class (currently not in use).
            :param dataset: The entire dataset (currently not in use).
        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        size_g_train = g.num_nodes()

        aux_train_ids = []

        if t!=self.current_task:
            # if new tasks come, store data from the current task
            self.current_task = t
            self.mix_g = g
            self.mix_train_ids = train_ids
            if t>0:
                last_task_sampled_ids = self.sampler(
                    self.last_task_ids_per_cls_train,
                    self.budget,
                    self.last_task_g,
                    self.net,
                    args,
                    self.last_task_offset1, self.last_task_offset2,
                    self.ce,
                    self.fai
                )
                old_ids = self.last_task_g.ndata['_ID'].cpu()
                self.buffer_node_ids.extend(old_ids[last_task_sampled_ids].tolist())
                aux_g, _, _ = dataset.get_graph(node_ids=self.buffer_node_ids, remove_edges=False)
                old_ids_aux = aux_g.ndata['_ID']
                aux_train_ids = [(old_ids_aux == i).nonzero().squeeze().item() + size_g_train for i in
                                self.buffer_node_ids]
                self.mix_g = dgl.batch([g,aux_g])
                self.mix_train_ids = np.concatenate((train_ids, aux_train_ids))
            self.last_task_g = g
            self.last_task_ids_per_cls_train = ids_per_cls_train
            self.last_task_offset1, self.last_task_offset2 = offset1, offset2

        dataloader = dgl.dataloading.NodeDataLoader(self.mix_g, self.mix_train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        # train in batches
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device('cuda:{}'.format(args.gpu))) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            loss.backward()
            self.opt.step()



