import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
from DataLoader import VideoQADataLoader_oie_bert_opendended_1

from utils import todevice
from utils import MultipleChoiceLoss
from validate import validate
import model.Model as MCL_ID
from utils_bert import todevice

from config_clip import cfg, cfg_from_file
from AutomaticWeightedLoss import AutomaticWeightedLoss
import warnings
import clip
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore")


def train_oie(cfg):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.train_question_pt,
        'question_subject_pt': cfg.dataset.train_question_subject_pt,
        'question_relation_pt': cfg.dataset.train_question_relation_pt,
        'question_object_pt': cfg.dataset.train_question_object_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'vocab_subject_json': cfg.dataset.vocab_subject_json,
        'vocab_relation_json': cfg.dataset.vocab_relation_json,
        'vocab_object_json': cfg.dataset.vocab_object_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'appearance_dict': cfg.dataset.appearance_dict,
        'motion_dict': cfg.dataset.motion_dict,
        'train_num': cfg.train.train_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': True
    }

    train_loader = VideoQADataLoader_oie_bert_opendended_1(**train_loader_kwargs)  # ,collate_fn=lambda x: x
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    if cfg.val.flag:
        val_loader_kwargs = {
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.val_question_pt,
            'question_subject_pt': cfg.dataset.val_question_subject_pt,
            'question_relation_pt': cfg.dataset.val_question_relation_pt,
            'question_object_pt': cfg.dataset.val_question_object_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'vocab_subject_json': cfg.dataset.vocab_subject_json,
            'vocab_relation_json': cfg.dataset.vocab_relation_json,
            'vocab_object_json': cfg.dataset.vocab_object_json,
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'appearance_dict': cfg.dataset.appearance_dict,
            'motion_dict': cfg.dataset.motion_dict,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False
        }
        val_loader = VideoQADataLoader_oie_bert_opendended_1(**val_loader_kwargs)
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'motion_dim': cfg.train.motion_dim,
        'appearance_dim': cfg.train.appearance_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        'vocab': train_loader.vocab,
        'question_type': cfg.dataset.question_type
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if
                           k != 'vocab' or k != 'vocab_subject' or k != 'vocab_relation' or k != 'vocab_object'}

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

    model = MCL_ID.MCL_ID(**model_kwargs).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('num of params: {}'.format(pytorch_total_params))

    if cfg.train.glove:
        logging.info('load glove vectors')
        train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)

        with torch.no_grad():
            model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)

    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0])

    optimizer = optim.Adam(model.parameters(), cfg.train.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0,
                           amsgrad=False)
    if cfg.dataset.question_type in ['none', 'frameqa']:
        criterion = nn.MultiMarginLoss().cuda()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    elif cfg.dataset.question_type in ['count']:
        criterion = nn.MSELoss().cuda()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    start_epoch = 0
    if cfg.dataset.question_type == 'count':
        best_val = 100.0
    else:
        best_val = 0
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    if cfg.dataset.question_type in ['frameqa', 'none']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.dataset.question_type == 'count':
        criterion = nn.MSELoss().to(device)
    logging.info("Start training........")
    best_acc = []
    train_time = []
    # awl = AutomaticWeightedLoss(4)
    loss_save = []
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        model.train()
        total_acc, count = 0, 0
        batch_mse_sum = 0.0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        train_accuracy = 0
        t1 = time.time()

        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]

            answers = answers.cuda().squeeze()
            batch_size = answers.size(0)
            optimizer.zero_grad()
            align_loss, logits = model(video_ids, question_ids,answers, clip_model, *batch_input)

            ans_loss = criterion(logits, answers)

            loss = 0.2*align_loss[0] +0.5*align_loss[1]+0.5*align_loss[2]+align_loss[3]+align_loss[4]
            
            (loss).backward()
            #scaler.scale(loss).backward()
            total_loss += loss.detach()
            avg_loss = total_loss / (i + 1)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            aggreeings = batch_accuracy(logits, answers)

            total_acc += aggreeings.sum().item()
            count += answers.size(0)
            train_accuracy = total_acc / count
            sys.stdout.write(
                "\rProgress = {progress}   loss = {loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}".format(
                    progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                    loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                    train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                      attrs=['bold']),
                    avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold'])))
            sys.stdout.flush()
            t2 = time.time()
            train_time.append(t2 - t1)

        loss_save.append(avg_loss.cpu().numpy())
        sys.stdout.write("\n")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        sys.stdout.flush()
        logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))

        if cfg.val.flag:
            t3 = time.time()
            output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                assert os.path.isdir(output_dir)
            valid_acc, *valid_output = validate(cfg, model, val_loader, device, write_preds=True)
            t4 = time.time()
            print("Average inference Time: {:.4f} secs".format(t4 - t3))
            scheduler.step(valid_acc)
            if (valid_acc > best_val and cfg.dataset.question_type != 'count') or (
                    valid_acc < best_val and cfg.dataset.question_type == 'count'):
                best_val = valid_acc
                ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                else:
                    assert os.path.isdir(ckpt_dir)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))
                preds_dir = os.path.join(cfg.dataset.save_dir, 'preds')
                preds = valid_output[0]
                preds = np.mat(preds)
                gts = valid_output[1]
                gts = np.mat(gts)
                v_ids = valid_output[2]
                v_ids = np.mat(v_ids)
                q_ids = valid_output[3]
                q_ids = np.mat(q_ids)

                np.savetxt(os.path.join(preds_dir, 'preds.csv'), preds, delimiter=",")
                np.savetxt(os.path.join(preds_dir, 'gts.csv'), gts, delimiter=",")
                np.savetxt(os.path.join(preds_dir, 'v_ids.csv'), v_ids, delimiter=",")
                np.savetxt(os.path.join(preds_dir, 'q_ids.csv'), q_ids, delimiter=",")
                sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                sys.stdout.flush()

            logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
            logging.info('~~~~~~ Valid Basic Accuracy: %.4f ~~~~~~~' % valid_output[4])
            logging.info('~~~~~~ Valid Attribution Accuracy: %.4f ~~~~~~' % valid_output[5])
            logging.info('~~~~~~ Valid Introspection Accuracy: %.4f ~~~~~~' % valid_output[6])
            logging.info('~~~~~~ Valid Counterfactual Accuracy: %.4f ~~~~~~' % valid_output[7])
            logging.info('~~~~~~ Valid Event Forecasting Accuracy: %.4f ~~~~~~' % valid_output[8])
            logging.info('~~~~~~ Valid Reverse Reasoning Accuracy: %.4f ~~~~~~' % valid_output[9])
            logging.info('~~~~~~ Valid Basic Count: %.4f ~~~~~~~' % valid_output[10])
            logging.info('~~~~~~ Valid Attribution Count: %.4f ~~~~~~' % valid_output[11])
            logging.info('~~~~~~ Valid Introspection Count: %.4f ~~~~~~' % valid_output[12])
            logging.info('~~~~~~ Valid Counterfactual Count: %.4f ~~~~~~' % valid_output[13])
            logging.info('~~~~~~ Valid Event Forecasting Count: %.4f ~~~~~~' % valid_output[14])
            logging.info('~~~~~~ Valid Reverse Reasoning Count: %.4f ~~~~~~' % valid_output[15])
            sys.stdout.write(
                '~~~~~~ Valid Accuracy: {valid_acc}, Basic Accuracy: {basic_acc}, Attribution Accuracy: {attri_acc}, Introspection Accuracy: {intro_acc}, Counterfactual Accuracy: {counter_acc}, Event Forecasting Accuracy: {forcast_acc}, Reverse Reasoning Accuracy: {reverse_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold']),
                    basic_acc=colored("{:.4f}".format(valid_output[4]), "red", attrs=['bold']),
                    attri_acc=colored('{:.4f}'.format(valid_output[5]), "red", attrs=['bold']),
                    intro_acc=colored('{:.4f}'.format(valid_output[6]), "red", attrs=['bold']),
                    counter_acc=colored('{:.4f}'.format(valid_output[7]), "red", attrs=['bold']),
                    forcast_acc=colored('{:.4f}'.format(valid_output[8]), "red", attrs=['bold']),
                    reverse_acc=colored('{:.4f}'.format(valid_output[9]), "red", attrs=['bold'])
                ))
            sys.stdout.flush()
        best_acc.append(valid_acc)

    best_epoch = max(best_acc)
    max_idx = best_acc.index(best_epoch)
    print("best epoch:", max_idx)
    loss_save1 = np.mat(loss_save)
    log_dir = os.path.join(cfg.dataset.save_dir, 'log')
    np.savetxt(os.path.join(log_dir, 'loss_save.csv'), loss_save1, delimiter=",")


# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        default='configs/sutd-qa.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['sutd-qa', 'tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)
    # check if k_max is set correctly
    # assert cfg.train.k_max_frame_level <= 16
    # assert cfg.train.k_max_clip_level <= 8

    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    # make logging.info display into both shell and file
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name,
                                                                                          cfg.dataset.question_type))
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name,
                                                                                      cfg.dataset.question_type))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name,
                                                                                                  cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.appearance_feat.format(cfg.dataset.name,
                                                                                      cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name,
                                                                                                    cfg.dataset.question_type))
    else:
        cfg.dataset.vocab_json = '{}_vocab_oie_bert.json'
        cfg.dataset.vocab_subject_json = '{}_vocab_oie_bert3.json'
        cfg.dataset.vocab_relation_json = '{}_vocab_oie_bert3.json'
        cfg.dataset.vocab_object_json = '{}_vocab_oie_bert3.json'
        cfg.dataset.train_question_pt = '{}_train_questions_bert_CF.pt'
        cfg.dataset.train_question_subject_pt = '{}_train_questions_subject_oie_bert3.pt'
        cfg.dataset.train_question_relation_pt = '{}_train_questions_relation_oie_bert3.pt'
        cfg.dataset.train_question_object_pt = '{}_train_questions_object_oie_bert3.pt'
        cfg.dataset.val_question_pt = '{}_test_questions_bert_CF.pt'  
        cfg.dataset.val_question_subject_pt = '{}_test_questions_subject_oie_bert3.pt'
        cfg.dataset.val_question_relation_pt = '{}_test_questions_relation_oie_bert3.pt'
        cfg.dataset.val_question_object_pt = '{}_test_questions_object_oie_bert3.pt'

        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name))
        cfg.dataset.train_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                             cfg.dataset.train_question_subject_pt.format(
                                                                 cfg.dataset.name))
        cfg.dataset.train_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                              cfg.dataset.train_question_relation_pt.format(
                                                                  cfg.dataset.name))
        cfg.dataset.train_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                            cfg.dataset.train_question_object_pt.format(
                                                                cfg.dataset.name))

        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                           cfg.dataset.val_question_subject_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                            cfg.dataset.val_question_relation_pt.format(
                                                                cfg.dataset.name))
        cfg.dataset.val_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                          cfg.dataset.val_question_object_pt.format(cfg.dataset.name))

        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))
        cfg.dataset.vocab_subject_json = os.path.join(cfg.dataset.data_dir,
                                                      cfg.dataset.vocab_subject_json.format(cfg.dataset.name))
        cfg.dataset.vocab_relation_json = os.path.join(cfg.dataset.data_dir,
                                                       cfg.dataset.vocab_relation_json.format(cfg.dataset.name))
        cfg.dataset.vocab_object_json = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.vocab_object_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))
        cfg.dataset.appearance_dict = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.appearance_dict.format(cfg.dataset.name))
        cfg.dataset.motion_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_dict.format(cfg.dataset.name))

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_oie(cfg)


if __name__ == '__main__':
    main()
