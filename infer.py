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

def get_predict(test_loader, net):
    load_iter = iter(test_loader)
    frame_predict = []
    
    if len(test_loader.dataset) == 1: # when there is a file in XD_Test.list
        _data, _label = next(load_iter)
        
        _data = _data.cuda()
        _label = _label.cuda()
        res = net(_data)   
        
        a_predict = res.cpu().numpy().mean(0)   

        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)
    else: # when in XD_Test.list there are five files (5-crop)
        for i in range(len(test_loader.dataset)//5):
            _data, _label = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()
            res = net(_data)   
            
            a_predict = res.cpu().numpy().mean(0)   

            fpre_ = np.repeat(a_predict, 16)
            frame_predict.append(fpre_)

    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict


def test(net, test_loader, model_file = None):
    st = time.time()
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))
        
        frame_predict = get_predict(test_loader, net)
        pred_binary = [1 if pred_value > 13.5 else 0 for pred_value in frame_predict]
        video_duration = int(np.ceil(len(pred_binary) * 0.96)) # len(pred_binary) = video_duration / 0.96
        print(pred_binary)
        if any(pred == 1 for pred in pred_binary):
            message= "El video contiene violencia. "
            message_second = "Los intervalos con violencia son: "
            message_frames = "En un rango de [0-"+ str(len(pred_binary) - 1) +"] los frames con violencia son: "

            start_idx = None
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    message_frames += ("[" + str(start_idx) + " - " + str(i - 1) + "]" + ", ") if i-1 != start_idx else ("[" + str(start_idx) + "], ")
                    message_second += ("[" + parse_time(int(np.floor((start_idx//16 + 1)* 0.96))) + " - " + parse_time(int(np.ceil(i//16 * 0.96))) + "], ")
                    start_idx = None

            if start_idx is not None:
                message_frames += ("[" + str(start_idx) + " - " + str(len(pred_binary) - 1) + "]") if len(pred_binary) - 1 != start_idx else ("[" + str(start_idx) + "]")
                message_second += ("[" + parse_time(int(np.floor((start_idx//16 + 1) * 0.96))) + " - " + parse_time(video_duration) + "]")
            else:
                message_frames = message_frames[:-2]              
                message_second = message_second[:-2]              

        else:
            message= "El video no contiene violencia."
            message_frames = ""            
            message_second = ""            

        # Create a list of dictionaries to store the data
        data = []
        data.append({
            'video_id': "IDVIDEO",
            'frame_number': pred_binary,
            "violence_label": "1" if any(pred == 1 for pred in pred_binary) else "0",
        })

        # Write the data to a CSV file
        csv_file = 'inference.csv'

        fieldnames = ['video_id', 'frame_number', 'violence_label']
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        time_elapsed = time.time() - st
        print(message + message_frames)
        print(message + message_second)
        print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    net = WSAD(args.len_feature, flag = "Test", args = args)
    net = net.cuda()
    test_loader = data.DataLoader(
        XDVideo(root_dir = args.root_dir, mode = 'Test', num_segments = args.num_segments, len_feature = args.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn)
    
    test(net, test_loader, model_file = args.model_path)