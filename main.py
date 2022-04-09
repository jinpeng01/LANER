
from src.config import get_params
from src.utils import init_experiment
from src.dataloader import get_dataloader, get_conll2003_dataloader, get_dataloader_for_bilstmtagger
from src.trainer import BaseTrainer
from src.model import BertTagger, BiLSTMTagger
from src.coach.dataloader import get_dataloader_for_coach
from src.coach.model import EntityPredictor
from src.coach.trainer import CoachTrainer

import torch
import numpy as np
from tqdm import tqdm
import random


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)


    if params.bilstm:
        # dataloader
        # dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        dataloader_train, dataloader_dev, dataloader_test = get_dataloader(params)

        # bilstm-crf model
        model = BiLSTMTagger(params)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)
    elif params.coach:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_coach(params)
        # coach model
        binary_tagger = BiLSTMTagger(params, vocab)
        entity_predictor = EntityPredictor(params)
        binary_tagger.cuda()
        entity_predictor.cuda()
        # trainer
        trainer = CoachTrainer(params, binary_tagger, entity_predictor)
    else:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test = get_dataloader(params)
        # BERT-based NER Tagger
        model = BertTagger(params)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('n_params', n_params)
    logger.info("*** paramaters ***" + str(n_params))


    if params.conll and not params.joint:
        conll_trainloader, conll_devloader, conll_testloader = get_conll2003_dataloader(params.batch_size, params.tgt_dm)
        trainer.train_conll(conll_trainloader, conll_devloader, conll_testloader, params.tgt_dm)



    no_improvement_num = 0
    best_f1 = 0
    logger.info("Training on target domain ...")
    dev_target = []
    test_target = []

    test_detail_results = []
    dev_detail_results = []

    for e in range(params.epoch):
        logger.info("============== epoch %d ==============" % e)


        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))

        # pbar = enumerate(dataloader_train)


        if params.bilstm:
            loss_list = []
            for i, (X, lengths, y) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss = trainer.train_step_for_bilstm(X, lengths, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

        elif params.coach:
            loss_bin_list, loss_entity_list = [], []
            for i, (X, lengths, y_bin, y_final) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss_bin, loss_entityname = trainer.train_step(X, lengths, y_bin, y_final)
                loss_bin_list.append(loss_bin)
                loss_entity_list.append(loss_entityname)
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f}; LOSS ENTITY:{:.4f}".format(e, np.mean(loss_bin_list),
                                                                                             np.mean(loss_entity_list)))

            logger.info("Finish training epoch %d. loss_bin: %.4f. loss_entity: %.4f" % (
            e, np.mean(loss_bin_list), np.mean(loss_entity_list)))

        else:
            loss_list = []
            for i, (X, y) in pbar:
                X, y = X.cuda(), y.cuda()
                loss = trainer.train_step(X, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

        logger.info("============== Evaluate epoch %d on Train Set ==============" % e)
        f1_train, report_results = trainer.evaluate(dataloader_train, params.tgt_dm, use_bilstm=params.bilstm)
        # logger.info("\n%s", report_results)

        logger.info("Evaluate on Train Set. F1: %.4f." % f1_train)

        logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
        f1_dev, report_results = trainer.evaluate(dataloader_dev, params.tgt_dm, use_bilstm=params.bilstm)
        # logger.info("\n%s", report_results)

        logger.info("Evaluate on Dev Set. F1: %.4f." % f1_dev)
        dev_target.append(f1_dev)
        dev_detail_results.append(report_results)

        logger.info("============== Evaluate epoch %d on Test Set ==============" % e)
        f1_test, report_results = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm)
        logger.info("\n%s", report_results)

        logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)
        test_target.append(f1_test)
        test_detail_results.append(report_results)

        if f1_dev > best_f1:
            logger.info("Found better model!!")
            best_f1 = f1_dev
            no_improvement_num = 0
            # trainer.save_model()
        else:
            no_improvement_num += 1
            logger.info("No better model found (%d/%d)" % (no_improvement_num, params.early_stop))


        if no_improvement_num >= params.early_stop:
            break

    dev_max_index = dev_target.index(max(dev_target))
    test_max_index = test_target.index(max(test_target))
    # import pdb
    # pdb.set_trace()

    logger.info("Best model basd on dev data.")
    logger.info("\n%s", test_detail_results[dev_max_index])

    logger.info("Best model basd on test data.")
    logger.info("\n%s", test_detail_results[test_max_index])

    logger.info("Best model basd on dev data. F1: %.4f." % test_target[dev_max_index])
    logger.info("Best model basd on test data. F1: %.4f." % max(test_target))



if __name__ == "__main__":
    params = get_params()

    random_seed(params.seed)
    train(params)
