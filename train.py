import copy
import argparse
from tqdm import trange

import torch
from torch_geometric.nn import GAE

from model import LinearEncoder, VariationalLinearEncoder
from data import Data


parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--dataset', type=str, default='dataset1',
                    choices=['dataset1', 'dataset2', 'dataset2'])
parser.add_argument('--out_channels', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--test_size', type=float, default=0.1)
args = parser.parse_args()

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data(args.dataset)
    
    if args.variational:
        model = VGAE(VariationalLinearEncoder(data.num_features, args.out_channels))
    else:
        model = GAE(LinearEncoder(data.num_features, args.out_channels))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    model = model.to(device)
    train_pos_edges, train_neg_edges, val_pos_edges, val_neg_edges = map(lambda x: x.to(device),
                                                                         data.split_edges(args.test_size))
    train_x = data.train_x.to(device)
    
    best_val_perf = 0
    _ = trange(args.epochs)

    for epoch in _:
        # training 
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_x, train_pos_edges)
        loss = model.recon_loss(z, train_pos_edges)
        if args.variational:
            loss = loss + (1 / train_x.size(0)) * model.kl_loss()
        loss.backward()
        optimizer.step()
        
        train_loss = float(loss)
        
        # vaildation
        model.eval()
        with torch.no_grad():
            z = model.encode(train_x, train_pos_edges)
            
        val_perf = model.test(z, val_pos_edges, val_neg_edges)[0]
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            best_model_params = copy.deepcopy(model.state_dict())

        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Best: {:.4f}'

        _.set_description(log.format(epoch, train_loss, val_perf, best_val_perf))
    
    model.load_state_dict(best_model_params)
    print('Best Performerce:', model.test(z, val_pos_edges, val_neg_edges)[0])
    
if __name__ == '__main__':
    main(args)
       