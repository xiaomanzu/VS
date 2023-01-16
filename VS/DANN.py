# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .eeg_seed import *

# from alg.modelopera import get_fea
from network import Adver_network, common_network
# from alg.algs.base import Algorithm
from network import img_network
parser.add_argument('--lr', '--learning-rate', default=4e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
def main():
    model=DANN()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)

    train_loader, train_givenY, train_sampler, test_loader = get_SEED_domain(target_domain, partial_rate=args.partial_rate,
                                                                         batch_size=args.batch_size)


def get_fea():
    net = nn.Sequential(
        nn.Linear(310, 256),
        # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Linear(256, 128),
        # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Linear(128, 64),
        # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )
    # if args.dataset == 'dg5':
    #     net = img_network.DTNBase()
    # elif args.net.startswith('res'):
    #     net = img_network.ResBase(args.net)
    # else:
    #     net = img_network.VGGBase(args.net)
    return net
class Algorithm(torch.nn.Module):

    def __init__(self, args):
        super(Algorithm, self).__init__()

    def update(self, minibatches, opt, sch):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
class DANN(Algorithm):

    def __init__(self, args):

        super(DANN, self).__init__(args)

        self.featurizer = get_fea()
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args = args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_z = self.featurizer(all_x)

        disc_input = all_z
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))