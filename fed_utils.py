import torch
import time, math
import utils
from scipy.sparse import coo_matrix
from utils import print2file, calc_f1, calc_f1_out
import numpy as np


def slave_run_train(model, args, package, pid="None"):
    model.train()
    t = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    list_loss, list_acc = [], []
    for iter in range(args.slave_ep):
        if args.dataset == "Amazon":
            feature = torch.tensor(package["features"], dtype=torch.float32)
        else:
            feature = torch.eye(len(package["features"]), dtype=torch.float32)
        support = to_torch_sparse_tensor(coo_matrix((package["support"][1], (package["support"][0][:, 0],
                                                                                package["support"][0][:, 1])),
                                                    shape=package["support"][2]))
        label = torch.tensor(package["y_train"], dtype=torch.float32)
        mask = torch.tensor(package["train_mask"].astype(int), dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss()

        if args.device >= 0:
            model = model.cuda()
            criterion = criterion.cuda()
            feature = feature.cuda()
            support = support.cuda()
            label = label.cuda().to(dtype=torch.int64)
            mask = mask.cuda()

        model.zero_grad()
        out = model(support, feature)

        loss, pred, acc = _metrics(out, label, mask, criterion, args.multilabel)

        # update model
        loss.backward()
        optimizer.step()

        # calculate F1 if needed.

        list_loss.append(loss.item())
        list_acc.append(acc.item())
        time_cost = time.time() - t

        # print(loss, acc)

    log_str = "Slave-" + str(pid) + " Done. Total time cost:" + str(time_cost) +\
              " average acc: " + str(sum(list_acc)/len(list_acc)) + ". average loss: " + \
              str(sum(list_loss) / len(list_loss))
    print2file(log_str, args.logDir, True)
    return {"params": model.cpu().state_dict(),
            "acc": sum(list_acc) / len(list_acc),
            "pred": pred,
            "out": out,
            "loss": sum(list_loss) / len(list_loss),
            "time": time_cost}


def slave_run_evaluate(model, args, package, pid="None"):

    model.eval()
    t = time.time()

    if args.dataset == "Amazon":
        feature = torch.tensor(package["features"], dtype=torch.float32)
    else:
        feature = torch.eye(len(package["features"]), dtype=torch.float32)
    support = to_torch_sparse_tensor(coo_matrix((package["support"][1], (package["support"][0][:, 0],
                                                                            package["support"][0][:, 1])),
                                                shape=package["support"][2]))
    label = torch.tensor(package["y_train"], dtype=torch.float32)
    mask = torch.tensor(package["train_mask"].astype(int), dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss()

    if args.device >= 0:
        model = model.cuda()
        criterion = criterion.cuda()
        feature = feature.cuda()
        support = support.cuda()
        label = label.cuda().to(dtype=torch.int64)
        mask = mask.cuda()

    out = model(support, feature)

    loss, pred, acc = _metrics(out, label, mask, criterion, args.multilabel)

    # list_loss.append(loss.item())
    # list_acc.append(acc.item())

    log_str = "Slave-" + str(pid) + " Done. Total time cost:" + str(time.time()-t) +\
              " average acc: " + str(acc.item()) + ". average loss: " + \
              str(loss.item())
    print2file(log_str, args.logDir, True)
    # print(log_str)
    return {"params": model.cpu().state_dict(),
            "acc": acc.item(),
            "pred": pred,
            "out": out,
            "loss": loss.item()}


def slave_run_train_AE(model, args, package, pid="None", train=True):
    t = time.time()
    if train:
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        model.eval()

    list_loss, list_acc = [], []

    for iter in range(args.slave_ep):

        if args.dataset == "Amazon":
            feature = torch.tensor(package["features"], dtype=torch.float32)
        else:
            feature = torch.eye(len(package["features"]), dtype=torch.float32)
        support = to_torch_sparse_tensor(coo_matrix((package["support"][1], (package["support"][0][:, 0],
                                                                                package["support"][0][:, 1])),
                                                    shape=package["support"][2])).to_dense()
        label = torch.tensor(package["y_train"], dtype=torch.float32)
        mask = torch.tensor(package["train_mask"].astype(int), dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss()

        if args.device >= 0:
            model = model.cuda()
            criterion = criterion.cuda()
            feature = feature.cuda()
            support = support.cuda()
            label = label.cuda().to(dtype=torch.int64)
            mask = mask.cuda()

        if train:
            model.zero_grad()
        embedding, recovered, out = model(support, feature)

        loss, pred, acc = _metrics(out, label, mask, criterion, args.multilabel)

        # loss = loss_function_AE(preds=recovered, labels=support, n_nodes=support.shape[0],
        #                         mask=mask)

        list_loss.append(loss.item())
        list_acc.append(acc.item())
        time_cost = time.time() - t

        if train:
            # update model
            loss.backward()
            optimizer.step()

        # calculate F1 if needed.

        list_loss.append(loss.item())
        # list_acc.append(acc.item())
        time_cost = time.time() - t

    log_str = "Slave-" + str(pid) + " Done. Total time cost:" + str(time_cost) + \
              " average acc: " + str(sum(list_acc) / len(list_acc)) + ". average loss: " + \
              str(sum(list_loss) / len(list_loss))
    print2file(log_str, args.logDir, True)

    return {"params": model.cpu().state_dict(),
            "acc": sum(list_acc) / len(list_acc),
            "pred": pred,
            "out": out,
            "loss": sum(list_loss) / len(list_loss),
            "time": time_cost}


def evaluate_AE(model, args, val_features_batches, val_support_batches,
             y_val_batches, val_mask_batches, val_data, pid="None"):
    """evaluate GCN model."""
    total_pred = []
    total_lab = []
    total_out = []
    total_loss = 0
    total_acc = 0

    num_batches = len(val_features_batches)
    for i in range(num_batches):

        features_b = val_features_batches[i]
        support_b = val_support_batches[i]
        y_val_b = y_val_batches[i]
        val_mask_b = val_mask_batches[i]
        num_data_b = np.sum(val_mask_b)

        if num_data_b == 0:
            continue
        else:
            package = {
                "features": features_b,
                "support": support_b,
                "y_train": y_val_b,
                "train_mask": val_mask_b
            }

            out_dict = slave_run_train_AE(model, args, package, pid=pid, train=False)

        total_pred.append(out_dict["pred"].cpu().detach().numpy()[val_mask_b])
        total_out.append(out_dict["out"].cpu().detach().numpy()[val_mask_b])
        total_lab.append(y_val_b[val_mask_b])

        # total_pred.append(out_dict["pred"][val_mask_b].cpu().tolist())
        # total_out.append(out_dict["out"][val_mask_b].cpu().tolist())
        # total_lab.append(y_val_b[val_mask_b])
        total_loss += out_dict["loss"] * num_data_b
        total_acc += out_dict["acc"] #* num_data_b

    total_pred = np.vstack(total_pred)
    total_out = np.vstack(total_out)
    total_lab = np.vstack(total_lab)
    loss = total_loss / len(val_data)
    acc = total_acc / num_batches

    micro, macro = calc_f1(total_pred, total_lab, args.multilabel)

    return loss, acc, micro, macro


# Define model evaluation function
def evaluate(model, args, val_features_batches, val_support_batches,
             y_val_batches, val_mask_batches, val_data, pid="None"):
    """evaluate GCN model."""
    total_pred = []
    total_lab = []
    total_out = []
    total_loss = 0
    total_acc = 0

    num_batches = len(val_features_batches)
    for i in range(num_batches):

        features_b = val_features_batches[i]
        support_b = val_support_batches[i]
        y_val_b = y_val_batches[i]
        val_mask_b = val_mask_batches[i]
        num_data_b = np.sum(val_mask_b)

        if num_data_b == 0:
            continue
        else:
            package = {
                "features": features_b,
                "support": support_b,
                "y_train": y_val_b,
                "train_mask": val_mask_b
            }

            out_dict = slave_run_evaluate(model, args, package, pid=pid)

        total_pred.append(out_dict["pred"].cpu().detach().numpy()[val_mask_b])
        total_out.append(out_dict["out"].cpu().detach().numpy()[val_mask_b])
        total_lab.append(y_val_b[val_mask_b])

        # total_pred.append(out_dict["pred"].cpu().tolist())
        # total_out.append(out_dict["out"].cpu().tolist())
        # total_lab.append(y_val_b[val_mask_b])

        # total_pred.append(out_dict["pred"][val_mask_b].cpu().tolist())
        # total_out.append(out_dict["out"][val_mask_b].cpu().tolist())
        # total_lab.append(y_val_b[val_mask_b])

        total_loss += out_dict["loss"] * num_data_b
        total_acc += out_dict["acc"] #* num_data_b

    total_pred = np.vstack(total_pred)
    total_out = np.vstack(total_out)
    total_lab = np.vstack(total_lab)
    loss = total_loss / len(val_data)
    acc = total_acc / num_batches

    micro, macro = calc_f1(total_pred, total_lab, args.multilabel)

    return loss, acc, micro, macro


def _loss_v2(preds, labels, mask, criterion, multilabel):
    """Construct the loss function."""
    loss_mask = mask.deepcopy()
    # Cross entropy error
    if multilabel:
        # loss_all = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=preds, labels=labels)
        loss_all = sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)      #shape[all_node_2993, 121]
        # mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        loss_mask /= torch.mean(loss_mask)
        # loss = tf.multiply(loss_all, mask[:, tf.newaxis])
        loss = loss_all * torch.unsqueeze(loss_mask, dim=1)
        # return tf.reduce_mean(loss)
        return torch.mean(loss)
    else:

        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        # mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        # loss *= mask
        # return tf.reduce_mean(loss)

        preds *= loss_mask
        labels *= loss_mask
        loss = criterion(preds, torch.max(labels, 1)[1])     #[NUM_NODE,1]

        return loss


def _metrics(outputs, labels, mask, criterion, multilabel):
    """Construct the loss function."""

    if multilabel:
        # loss
        loss_all = sigmoid_cross_entropy_with_logits(logits=outputs, labels=labels)
        mask /= torch.mean(mask)
        loss = loss_all * torch.unsqueeze(mask, dim=1)
        # pred
        pred = torch.sigmoid(outputs)
        # acc
        outputs = outputs > 0
        labels = labels > 0.5
        accuracy_all = torch.eq(outputs, labels).to(dtype=torch.float32)
        accuracy_all = accuracy_all * torch.unsqueeze(mask, dim=1)

        return torch.mean(loss), pred, torch.mean(accuracy_all)

    else:
        mask = mask.to(dtype=torch.bool)
        outputs = outputs[mask]
        labels = labels[mask]
        loss = criterion(outputs, torch.max(labels, 1)[1])
        # pred
        pred = torch.nn.functional.softmax(outputs, dim=-1)
        # acc
        _, c_indices = torch.max(pred, 1)
        _, l_indices = torch.max(labels, 1)
        correct_prediction = torch.eq(c_indices, l_indices)
        accuracy_all = correct_prediction.to(dtype=torch.float32)
        # mask /= torch.mean(mask)
        # accuracy_all *= mask

        return loss, pred, torch.mean(accuracy_all)


def _predict(outputs, multilabel):
    if multilabel:
        pred = torch.sigmoid(outputs)
    else:
        pred = torch.nn.functional.softmax(outputs, dim=-1)

    return pred


def _accuracy(outputs, labels, labels_mask, multilabel):
    acc_mask = labels_mask.deepcopy()
    if multilabel:
        # accuracy = utils.masked_accuracy_multilabel(outputs, labels,labels_mask)

        # preds = preds > 0
        # labels = labels > 0.5
        outputs = outputs > 0
        labels = labels > 0.5
        # correct_prediction = tf.equal(preds, labels)
        # accuracy_all = tf.cast(correct_prediction, tf.float32)
        accuracy_all = torch.eq(outputs, labels).to(dtype=torch.float32)
        # mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        acc_mask /= torch.mean(acc_mask)
        # accuracy_all = tf.multiply(accuracy_all, mask[:, tf.newaxis])
        accuracy_all = accuracy_all * torch.unsqueeze(acc_mask, dim=1)

        # return tf.reduce_mean(accuracy_all)
        return torch.mean(accuracy_all)
    else:
        # accuracy = utils.masked_accuracy(outputs, labels, labels_mask)

        # correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        _, c_indices = torch.max(outputs, 1)
        _, l_indices = torch.max(labels, 1)
        correct_prediction = torch.eq(c_indices, l_indices)
        # accuracy_all = tf.cast(correct_prediction, tf.float32)
        accuracy_all = correct_prediction.to(dtype=torch.float32)
        # mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        acc_mask /= torch.mean(acc_mask)
        accuracy_all *= acc_mask
        return torch.mean(accuracy_all)


def to_torch_sparse_tensor(coo):
    """Convert Scipy sparse matrix to torch sparse tensor."""
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
    """
    pytorch version of tf.nn.sigmoid_cross_entropy_with_logits
    Implementation based on https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    :param labels:
    :param logits:
    :return:
    """
    # logits[logits < 0] = 0
    return torch.nn.functional.relu(logits) - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))


def loss_function_AE(preds, labels, n_nodes, mask):

    # cost = torch.nn.functional.binary_cross_entropy_with_logits(preds, labels)
    # cost = norm * tf.reduce_mean(
    #     tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
    # labels = labels.to(dtype=torch.int64)
    cost = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(preds, labels))
    # cost = torch.mean(torch.pow(labels-preds, 2))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = 0.5 * torch.sum()
    # KLD = -0.5 / n_nodes * torch.mean(torch.sum(
    #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost
