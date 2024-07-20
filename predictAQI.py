import os

from models.ChaIR import build_net
import torch
from data import test_dataloader2, dataloadsvm
import torch.nn.functional as f
import json
import argparse
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from torchvision.transforms import functional as F
import pickle


def predict_AQI(args):
    model = build_net()
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader2(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()

    rs = []

    factor = 4
    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            input_check = input_img.clone()

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]
            # print(input_img[:, :, :h, :w])
            # print(pred.shape)
            pred = torch.clamp(pred, 0, 1)
            # label_img = label_img.to(device)
            loss = f.l1_loss(pred, input_check).item()
            pred = F.to_pil_image(pred.squeeze(0).cpu(), 'RGB')
            input_check = F.to_pil_image(input_check.squeeze(0).cpu(), 'RGB')
            pred.save(os.path.join("results", "testAQI", "pred" + name[0]))
            input_check.save(os.path.join("results", "testAQI", "input" + name[0]))
            type = int(name[0].split('.')[0][-2:])
            label_aq = 1
            if type == 15:
                label_aq = 2
            if type == 20:
                label_aq = 3
            if type == 25:
                label_aq = 4
            if type == 30:
                label_aq = 5

            rs.append((name, loss, label_aq))
    return rs


def train_svm(args):
    model = build_net()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()

    trainloader = dataloadsvm(args.data_dir, batch_size=1, num_workers=0, phase="train")
    validloader = dataloadsvm(args.data_dir, batch_size=1, num_workers=0, phase="valid")

    factor = 4
    rstrain = []
    rsvalid = []

    with torch.no_grad():
        for iter_idx, data in enumerate(trainloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            input_check = input_img.clone()

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]
            pred = torch.clamp(pred, 0, 1)
            # label_img = label_img.to(device)
            loss = f.l1_loss(pred, input_check).item()

            if args.save_image:
                pred = F.to_pil_image(pred.squeeze(0).cpu(), 'RGB')
                input_check = F.to_pil_image(input_check.squeeze(0).cpu(), 'RGB')
                pred.save(os.path.join(args.result_dir, "image", "pred" + name[0]))
                input_check.save(os.path.join(args.result_dir, "image", "input" + name[0]))
            type = int(name[0].split('.')[0][-2:])
            label_aq = 1
            if type == 15:
                label_aq = 2
            if type == 20:
                label_aq = 3
            if type == 25:
                label_aq = 4
            if type == 30:
                label_aq = 5

            rstrain.append((loss, label_aq))

        for iter_idx, data in enumerate(validloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            input_check = input_img.clone()

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]
            pred = torch.clamp(pred, 0, 1)
            # label_img = label_img.to(device)
            loss_kl = f.kl_div(pred, input_check).item()
            loss_l1 = f.l1_loss(pred, input_check).item()
            loss_cross = f.cross_entropy(pred, input_check).item()
            # loss_gau_ne = f.gaussian_nll_loss(pred, input_check).item()

            if args.save_image:
                pred = F.to_pil_image(pred.squeeze(0).cpu(), 'RGB')
                input_check = F.to_pil_image(input_check.squeeze(0).cpu(), 'RGB')
                pred.save(os.path.join(args.result_dir, "image", "pred" + name[0]))
                input_check.save(os.path.join(args.result_dir, "image", "input" + name[0]))
            type = int(name[0].split('.')[0][-2:])
            label_aq = 1
            if type == 15:
                label_aq = 2
            if type == 20:
                label_aq = 3
            if type == 25:
                label_aq = 4
            if type == 30:
                label_aq = 5

            rsvalid.append((loss_l1, label_aq))

    rstrain = np.array(rstrain)
    rsvalid = np.array(rsvalid)

    X_train, y_train, X_valid, y_valid = rstrain[:, 0].reshape(-1, 1), rstrain[:, 1], rsvalid[:, 0].reshape(-1, 1), rsvalid[:, 1]

    clf = SVC(gamma="auto", kernel="linear")
    clf.fit(X_train, y_train)

    with open(os.path.join(args.result_dir, "clf", "classifier.pkl"), "wb") as fil:
        pickle.dump(clf, fil, protocol=5)

    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_valid, y_valid)

    return score_train, score_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results', 'testAQI')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    path_image = os.path.join(args.result_dir, "image")
    if not os.path.exists(path_image):
        os.makedirs(path_image)
    path_clf = os.path.join(args.result_dir, "clf")
    if not os.path.exists(path_clf):
        os.makedirs(path_clf)

    print(args)
    if args.mode == "train":
        score_train, score_test = train_svm(args)
        print("Score_train: ", score_train)
        print("Score_test: ", score_test)
    else:
        rs = predict_AQI(args)
        with open("rs.json", 'w') as file:
            json.dump(rs, file)
