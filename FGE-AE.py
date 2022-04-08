import os
from parser import parameter_parser
import partition_utils
import numpy as np
import utils
import torch
from layers import GCNModelAE
from fed_utils import slave_run_train_AE, evaluate_AE
import copy
from aggregate import average_agg, weighted_agg
from utils import print2file
import pickle
import time


def main():
    print("Program start, environment initializing ...")
    # torch.autograd.set_detect_anomaly(True)
    args = parameter_parser()
    utils.print2file(str(args), args.logDir, True)

    if args.device >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pic = {}

    # check if pickles, otherwise load data
    pickle_name = args.data_prefix+args.dataset+"-"+str(args.bsize)+str(args.num_clusters)+str(args.precalc)+\
                                                                                           "_FGE-AE"+".pickle"
    if os.path.isfile(pickle_name):
        print("Loading Pickle.")
        load_time = time.time()
        pic = pickle.load(open(pickle_name, "rb"))
        print("Loading Done. " + str(time.time()-load_time) + " seconds.")
    else:
        print("Data Pre-processing")
        # Load data
        (pic["train_adj"], full_adj, pic["train_feats"], pic["test_feats"], pic["y_train"], y_val, y_test,
         pic["train_mask"], pic["val_mask"], test_mask, _, pic["val_data"], pic["test_data"], num_data,
         visible_data) = utils.load_data(args.data_prefix, args.dataset, args.precalc)

        print("Partition graph and do preprocessing")
        if args.bsize > 1:
            _, pic["parts"] = partition_utils.partition_graph(pic["train_adj"], visible_data,
                                                       args.num_clusters)
            pic["parts"] = [np.array(pt) for pt in pic["parts"]]

            (pic["features_batches"], pic["support_batches"], pic["y_train_batches"],
             pic["train_mask_batches"]) = utils.preprocess_multicluster_v2(
                pic["train_adj"], pic["parts"], pic["train_feats"], pic["y_train"], pic["train_mask"],
                args.num_clusters, args.bsize, args.diag_lambda)

        else:
            (pic["parts"], pic["features_batches"], pic["support_batches"], pic["y_train_batches"],
             pic["train_mask_batches"]) = utils.preprocess(pic["train_adj"], pic["train_feats"], pic["y_train"],
                                                    pic["train_mask"], visible_data,
                                                    args.num_clusters,
                                                    args.diag_lambda)

        (_, pic["val_features_batches"], pic["val_support_batches"], pic["y_val_batches"],
         pic["val_mask_batches"]) = utils.preprocess(full_adj, pic["test_feats"], y_val, pic["val_mask"],
                                              np.arange(num_data),
                                              args.num_clusters_val,
                                              args.diag_lambda)

        (_, pic["test_features_batches"], pic["test_support_batches"], pic["y_test_batches"],
         pic["test_mask_batches"]) = utils.preprocess(full_adj, pic["test_feats"], y_test,
                                               test_mask, np.arange(num_data),
                                               args.num_clusters_test,
                                               args.diag_lambda)

        pickle.dump(pic, open(pickle_name, "wb"))

    idx_parts = list(range(len(pic["parts"])))
    print("Preparing model ...")
    model = GCNModelAE(pic["test_feats"].shape[1], args.hidden1, args.hidden1, pic["y_train"].shape[1],
                        args.dropout)

    w_server = model.cpu().state_dict()

    print("Start training ...")
    model_saved = "./model/" + args.dataset + "-" + args.logDir[6:-4] + ".pt"

    try:
        for epoch in range(args.epochs):
            # Training process
            w_locals, loss_locals = [], []
            all_time = []

            # for pid in range(len(pic["features_batches"])):
            for pid in range(10):
                # Use preprocessed batch data
                package = {
                    "features": pic["features_batches"][pid],
                    "support": pic["support_batches"][pid],
                    "y_train": pic["y_train_batches"][pid],
                    "train_mask": pic["train_mask_batches"][pid]
                }

                model.load_state_dict(w_server)
                out_dict = slave_run_train_AE(model, args, package, pid)

                w_locals.append(copy.deepcopy(out_dict['params']))
                loss_locals.append(copy.deepcopy(out_dict['loss']))
                all_time.append(out_dict["time"])

            # update global weights
            a_start_time = time.time()
            if args.agg == 'avg':
                w_server = average_agg(w_locals, args.dp)
            elif args.agg == 'att':
                w_server = weighted_agg(w_locals, w_server, args.epsilon, args.ord, dp=args.dp)
            else:
                exit('Unrecognized aggregation')

            model.load_state_dict(w_server)
            # agg_time = time.time() - a_start_time
            # print(str(sum(all_time)/len(all_time) + agg_time))
            print2file('Epoch: ' + str(epoch) + ' Train acc: ' + str(out_dict["acc"]), args.logDir, True)

            if epoch % args.val_freq == 0:
                best_val_acc = 0
                best_epoch = 0
                val_cost, val_acc, val_micro, val_macro = evaluate_AE(model, args,
                                                                   pic["val_features_batches"],
                                                                   pic["val_support_batches"],
                                                                   pic["y_val_batches"],
                                                                   pic["val_mask_batches"],
                                                                   pic["val_data"],
                                                                   pid="validation")

                log_str = 'Validateion set results: ' + 'cost= {:.5f} '.format(
                    val_cost) + 'accuracy= {:.5f} '.format(
                    val_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(val_micro, val_macro)
                print2file(log_str, args.logDir, True)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_saved)
                    print2file("Best val_acc: " + str(best_val_acc) + " with epoch: " + str(best_epoch), args.logDir,
                               True)

        # Test Model
        # Perform two test, one with last model, another with best val_acc model
        # 1)
        test_cost, test_acc, micro, macro = evaluate_AE(model, args, pic["test_features_batches"],
                                                     pic["test_support_batches"], pic["y_test_batches"],
                                                     pic["test_mask_batches"], pic["test_data"], pid="Final test")

        log_str = 'Test set results: ' + 'cost= {:.5f} '.format(
            test_cost) + 'accuracy= {:.5f} '.format(
            test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
        print2file(log_str, args.logDir, True)


        # 2)
        test_model = GCNModelAE(pic["test_feats"].shape[1], args.hidden1, args.hidden1,
                            args.dropout, precalc=args.precalc)

        test_model.load_state_dict(torch.load(model_saved))
        test_model.eval()
        test_cost, test_acc, micro, macro = evaluate_AE(test_model, args, pic["test_features_batches"],
                                                     pic["test_support_batches"], pic["y_test_batches"],
                                                     pic["test_mask_batches"], pic["test_data"], pid="Best test")


        log_str = 'Test set results: ' + 'cost= {:.5f} '.format(
            test_cost) + 'accuracy= {:.5f} '.format(
            test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
        print2file(log_str, args.logDir, True)


    except KeyboardInterrupt:
        print("==" * 20)
        print("Existing from training earlier than the plan.")

    print("End..so far so good.")


if __name__ == "__main__":
    main()
