import torch
import numpy as np
from dataset_loader import XDVideo
from options import parse_args
import pdb
import utils
from models import WSAD
from dataset_loader import data
import time
import csv
import os

def get_predict(test_loader, net):
    load_iter = iter(test_loader)
    frame_predict = []
    
    for i in range(len(test_loader.dataset)):
        _data, _label = next(load_iter)
        
        _data = _data.to(device)
        _label = _label.to(device)
        res = net(_data)   
        
        a_predict = res.cpu().numpy().mean(0)   

        frame_predict.append(a_predict)

    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict


def test(net, test_loader, args, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        
        frame_predict = get_predict(test_loader, net)
        pred_binary = [1 if pred_value > 13.5 else 0 for pred_value in frame_predict]
        return pred_binary

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec

def save_results(results, filename):
    np.save(filename, results)

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    net = WSAD(args.len_feature, flag="Test", args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    test_loader = data.DataLoader(
        XDVideo(root_dir=args.rgb_list, mode='Test', num_segments=args.num_segments, len_feature=args.len_feature),
        batch_size=1,
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn
    )
    
    results = test(net, test_loader, args, model_file=args.model_path)
    save_results(results, os.path.join(args.output_path, 'results.npy'))
