import logging
import numpy as np
import math
import random
def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger



def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
    logger.info(output)

    return

def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1-test_prop) * n_all).astype(int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor(test_prop * n_all).astype(int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx

def cal_HAR(logger, *arg):
    """ print classification results for HAR """
    if len(arg) == 5:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        logger.info(arg[3])
        logger.info(arg[4])
        output = """ 
                     RGB {:.2f} std {:.2f}
                     Depth {:.2f} std {:.2f} 
                     RGB+D {:.2f} std {:.2f}
                     onlyrgb {:.2f} std {:.2f}
                     onlydepth {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100, np.std(arg[1]) * 100,
                                                           np.mean(arg[2]) * 100, np.std(arg[2]) * 100,
                                                           np.mean(arg[3]) * 100, np.std(arg[3]) * 100,
                                                           np.mean(arg[4]) * 100, np.std(arg[4]) * 100)

        logger.info(output)
    return


def cal_classify(logger, *arg):
    """ print classification results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.2f} std {:.2f}
                     Precision {:.2f} std {:.2f} 
                     F-measure {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100,
                                                           np.std(arg[1]) * 100, np.mean(arg[2]) * 100,
                                                           np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)
    return