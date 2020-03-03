import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

"""
利用哈工大的自然语言处理工具实现：分词，词性标注，依存句法分析，命名体识别，语义角色识别

roles_dict[role.index] = {arg.name:[arg.name, arg.range.start, arg.range.end] for arg in role.arguments}  # 这也叫列表生成式？
"""
class SentenceParser:
    def __init__(self):
        LTP_DIR = './ltp_data_v3.4.0'
        print(LTP_DIR)
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))

        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl.model'))


    '''句法分析---为句子中的每个词语维护一个依存句法依存儿子节点（词的出度）的字典'''
    '''
        句法分析中，每个只有一个入度（可能吧），可能有多个出度。
        为了可以结构化的展示分析结果，或者说方便提取信息。
        对每个词建立一个子节点的字典：
            1) 若该词的出度为0，字典为NULL
            2) 若该词的出度为n，那字典的元素个数为n
    '''
    def build_parse_child_dict(self, words, postags, arcs):
        """
        格式化句法分析结果
        :param words: 分词结果
        :param postags: 词性标注结果
        :param arcs: 句法分析结果
        :return: child_dict_list, format_parse_list
        """
        '''
        arcs是一个列表：
            列表元素当前单词，每个元素arc包含arc.head, arc.relation信息，
            head为指向该词（词的父节点）的下标（从1开始），relation为父节点和该词的句法关系
            *** 因为每个词只有 一个入度， 这个arc信息就表示入度信息
            
        LTP句法分析模型输出arcs：表示每个词的入度信息，父节点信息，只有一个
        返回：
            child_dict_list：是表示每个词的出度信息，就是子节点信息
            format_parse_list：每个词信息格式化：  与父节点句法关系，该词，该词下标，该词词性，父节点词，父词下标，父词词性
        '''

        child_dict_list = []
        format_parse_list = []

        # 对每个词建立子节点信息
        for index in range(len(words)):
            child_dict = dict()
            ## 遍历寻找该词的子节点
            for arc_index in range(len(arcs)):
                ## 如果有指向该词的子节点，则加入child_dict
                if arcs[arc_index].head == index+1:
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)

            child_dict_list.append(child_dict)


        # 对每个词建立指定信息
        ## 包含: [依存关系，词，下标，POS，父节点词，父节点下标，父节点POS]  # 还可以加上词的NER信息
        rely_id = [arc.head for arc in arcs]  # 提取每个词依存父节点id（其中id为0的是Root）
        relation = [arc.relation for arc in arcs]  # 提取每个词依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list




    '''语义角色标注'''
    '''
        只对句子中 谓词 进行论元分析，抽取论元以及标注论元和谓词的关系。
    '''
    def format_labelrole(self, words, postags):
        """
        格式化语义角色标注结果
        :param words:
        :param postags:
        :return:
        """
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        '''
        roles中有多个role，每个role代表句子中的一个谓词
            role.index 代表谓词的索引， 
            role.arguments 代表关于该谓词的若干语义角色。（这里的论元可能不是简单的一个词）
                arg.name 表示语义角色类型，
                arg.range.start 表示该语义角色起始词位置的索引，(索引从0开始）
                arg.range.end 表示该语义角色结束词位置的索引。
        roles={
            'r1':{
                'args1':{
                    'name': 语义角色类型,
                    'range':{
                        'start': 语义角色起始词位置的索引,
                        'end': 语义角色结束词位置的索引
                    }
                },
                'args2':{
                    'name': 语义角色类型,
                    'range': {
                        'start': 语义角色起始词位置的索引,
                        'end': 语义角色结束词位置的索引
                    }
                },
                ...
            },
            'r2':{
                'args1': {
                    'name': 语义角色类型,
                    'range': {
                        'start': 语义角色起始词位置的索引,
                        'end': 语义角色结束词位置的索引
                    }
                },
                'args2': {
                    'name': 语义角色类型,
                    'range': {
                        'start': 语义角色起始词位置的索引,
                        'end': 语义角色结束词位置的索引
                    }
                },
                ...
            },
            ...
        }
        '''
        for role in roles:
            roles_dict[role.index] = {arg.name: [arg.name, arg.range.start, arg.range.end] for arg in role.arguments}
        return roles_dict






    '''parser主函数'''
    '''
    将模型的输出进行处理，方便之后数据处理
        模型输出：words, postags, ners, arcs, roles
        处理后信息：
            child_dict_list：句法分析，每个词的子节点信息
            format_parse_list：句法分析，每个词的信息和父节点信心（父节点唯一）
            roles_dic：
    '''
    def parser_main(self, sentence):

        '''words, postags, ners, arcs 为LTP模型输出'''
        words = list(self.segmentor.segment(sentence))
        postags = list(self.postagger.postag(words))
        ners = list(self.recognizer.recognize(words, postags))
        arcs = self.parser.parse(words, postags)

        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        """
        arcs中有多个arc
            arc.head 表示依存弧的父节点词的索引。ROOT节点的索引是0，第一个词开始的索引依次为1、2、3…
            arc.relation 表示依存弧的关系。
            注意：一个词最多只有一个弧指向它（即只有一个入度），但是一个词可以指向多个词（即有多个出度）
        """
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(words, postags)

        return words, postags, ners, child_dict_list, format_parse_list, roles_dict


if __name__ == '__main__':
    parse = SentenceParser()
    sentence = '李克强总理今天来我家了,我感到非常荣幸'
    words, postags, ners, child_dict_list, format_parse_list, roles_dict = parse.parser_main(sentence)
    print(words)
    import json
    print("-------------------每个词在句法分析树上的所有子节点信息----------------------")
    print(json.dumps(child_dict_list,indent=4, ensure_ascii=False))

    print("\n-------------------每个句法分析中词信息和父节点信息----------------------")
    print(json.dumps(format_parse_list,indent=4, ensure_ascii=False))

    print("-------------------句子中所有语义角色信息（即每个谓词的语义角色信息）----------------------")
    print(json.dumps(roles_dict,indent=4, ensure_ascii=False))
