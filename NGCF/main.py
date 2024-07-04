import torch
import torch.optim as optim
import os

from NGCF import NGCF
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
 
if __name__ == '__main__':

    args.device = torch.device("cpu")

    plain_adj, norm_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    if args.pretrain:
        # Specify the path to the weights file
        weights_file = os.path.join(args.weights_path, '379.pkl')
        
        # Check if the weights file exists
        if os.path.exists(weights_file):
            # Load the weights from the specified file
            model.load_state_dict(torch.load(weights_file))

            # Evaluate the pretrained model
            users_to_test = list(data_generator.test_set.keys())
            ret = test(model, users_to_test, drop_flag=False)
            cur_best_pre_0 = ret['recall'][0]

            # Print the evaluation results of the pretrained model
            pretrained_results = 'recall=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % (ret['recall'][0], ret['recall'][-1], ret['ndcg'][0], ret['ndcg'][-1])
            print(f'Pretrained model (Epoch 399):', pretrained_results)
        else:
            # Exit if the specified weights file is not found
            print('Specified weights file not found')
            exit(1)
    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # print("Training step ",idx," Loss: ",batch_loss)
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            if os.path.isdir("./model"):
                torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
                print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
            else:
                os.mkdir("./model")
                torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
                print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)