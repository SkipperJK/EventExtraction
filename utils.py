import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG


def get_ltp_entity(word_list, arcs):
    # 拆分成三个计算的时候要for循环三次，会不会影响计算速度，有必要拆开吗，效率和可读性之间的tradeoff
    who = get_who_entity(word_list, arcs)
    whom = get_whom_entity(word_list, arcs)
    predicate = get_predicate_entity(word_list, arcs)
    return who, whom, predicate


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-PER':
                if 'per' in locals().keys():
                    PER.append(per)
                    del per
                per = char
                if i+1 == length:
                    PER.append(per)
            if tag == 'I-PER':
                per += char
                if i+1 == length:
                    PER.append(per)
            if tag not in ['I-PER', 'B-PER']:
                if 'per' in locals().keys():
                    PER.append(per)
                    del per
                continue
    except:
        pass
    #     PER = []
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-LOC':
                if 'loc' in locals().keys():
                    LOC.append(loc)
                    del loc
                loc = char
                if i+1 == length:
                    LOC.append(loc)
            if tag == 'I-LOC':
                loc += char
                if i+1 == length:
                    LOC.append(loc)
            if tag not in ['I-LOC', 'B-LOC']:
                if 'loc' in locals().keys():
                    LOC.append(loc)
                    del loc
                continue
    except:
        pass
    #     LOC = []
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []

    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-ORG':
                if 'org' in locals().keys():
                    ORG.append(org)
                    del org
                org = char
                if i+1 == length:
                    ORG.append(org)
            if tag == 'I-ORG':
                org += char
                if i+1 == length:
                    ORG.append(org)
            if tag not in ['I-ORG', 'B-ORG']:
                if 'org' in locals().keys():
                    ORG.append(org)
                    del org
                continue
    except:
        pass
    #     ORG = []
    return ORG


def get_who_entity(word_list, arcs):
    who = []
    for i, arc in enumerate(arcs):
        if arc.relation == "SBV":  # subject verb
            who.append(word_list[i])
    return who

def get_whom_entity(word_list, arcs):
    whom = []
    for i, arc in enumerate(arcs):
        if arc.relation == "VOB":  # object verb
            whom.append(word_list[i])
    return whom


def get_predicate_entity(word_list, arcs):
    predicate =[]
    for i, arc in enumerate(arcs):
        if arc.relation == "HED":
            predicate.append(word_list[i])
    return predicate


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
