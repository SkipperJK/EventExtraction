from TripleExtract.sentence_parser import *
from TripleExtract.utlis import *
import re
import json

"""事件模版提取
想要的是通过从语料库中抽取大量的三元组信息，挖掘事件模版。
    三元组：[sub, verb, obj]
    考虑一个问题：对于事件来说，不一定有obj，例如：xxx怀孕，但是一定有sub吗？那抽取的时候，不仅要保留完整的三元组信息还要保留不完整的[sub, verb, '']

通过分析发现：如果从正文中提取，可能会提到很多无关紧要的三元组，按照我们的想法，提取事件模版，想的是**已知**这个时间段发生了什么，通过对这个时间段新闻的分析，找出发生该事件的模版。那么要不要只对文章的标题进行抽取。
TO-DO： 应该可以通过目前深度学习进行分词，词性标注，等等得到的信息，转换成LTP的标注方法，使用后续的DP和SRL技术

**** 信息抽取中的SPO抽取，指的是 实体-关系-实体， 实体-属性-实体，
但是这里在句法分析上进行的抽取是基于动词的抽取，不是指定多少中关系。

看到有的是吧SPO抽取看成一个多分类问题，就是设定好实体间多少中关系的类别，只抽取这么多种的关系。


实现方法： 
    通过对LTP的各个模型输出处理之后（即sentence_parser程序做的预处理），提取句子中的三元组信息
"""




class TripleExtractor:
    def __init__(self):
        self.parser = SentenceParser()


    # 分句函数一
    def split_sents(self, content):
        return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]

    # 分句函数二
    def cut_sent(self, content):
        content = re.sub('([。！？\?])([^”’])', r"\1\n\2", content)  # 单字符断句符
        content = re.sub('(\.{6})([^”’])', r"\1\n\2", content)  # 英文省略号
        content = re.sub('(\…{2})([^”’])', r"\1\n\2", content)  # 中文省略号
        content = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', content)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        content = content.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return content.split("\n")



    '''
    利用语义角色标注,直接获取主谓宾三元组,
        每个元素都是动词，以及与该动词相关的语义信息： A0, A1
    '''
    def according_srl_extract(self, words, postags, nertags, roles_dict, role_index):
        """
        根据动词下标role_index，和roles_dict找出给动词的三元组
        :param words:
        :param postags:
        :param nertags:
        :param roles_dict: roles_dict[role.index] = {arg.name:[arg.name,arg.range.start, arg.range.end] for arg in role.arguments}
        :param role_index: 某个动词的下标
        :return:  返回输入动词对应的三元组信息
        """
        # ------------------ 加上TMP和LOC就不能叫做triple了 --------------------
        triple = {
            'sub': {},
            'verb': '',
            'obj': {}
        }

        triple['verb'] = words[role_index]
        role_info = roles_dict[role_index]

        # 能否用一个谓词的语义角色标注来实现事件提取？？？？
        # 因为语义角色标注就是实现，动词，动词的主语和宾语，还有与该动词相关的时间词和地点词等等
        ## 提取谓词对应的A0和A1论元
        ### 由于论元由多个词组成，去掉词POS不是 wp,ws,x,u的词
        ### *** 这里如果把NER信息传递过来，也可以通过NER来过滤词 ***
        ### 记录每个词的POS和NER信息（目前只能使用LTP得到的POS和NER标注信息）

        # A0：动作的施事（宾语） A1：动作的影响（这里当作宾语）
        if 'A0' in role_info.keys():
            # triple['sub']['words'], triple['sub']['postags'] = get_WordsAndPostagsOfRole(words, postags, role_info['A0'][1], role_info['A0'][2])
            triple['sub'] = get_WordsDictOfRole(words, postags, role_info['A0'][1], role_info['A0'][2])


        if 'A1' in role_info.keys():
            triple['obj'] = get_WordsDictOfRole(words, postags, role_info['A1'][1], role_info['A1'][2])


        if 'LOC' in role_info.keys():
            triple['loc'] = {}
            triple['loc'] = get_WordsDictOfRole(words, postags, role_info['LOC'][1], role_info['LOC'][2])


        if 'TMP' in role_info.keys():
            triple['tmp'] = {}
            triple['tmp'] = get_WordsDictOfRole(words, postags, role_info['TMP'][1], role_info['TMP'][2])

        if 'A0' in role_info.keys() and 'A1' in role_info.keys():
            return triple
        else:
            return {}


    '''
    利用依存句法分析提取三元组
    '''
    def accroding_dp_extarct(self, words, postags, nertags, child_dict_list, format_parse_list, word_index):
        flag = False
        triple = {
            'sub': {},
            'verb': '',
            'obj': {}
        }

        triple['verb'] = words[word_index]
        child_dict = child_dict_list[word_index]


        if 'SBV' in child_dict:
            triple['sub'][words[child_dict['SBV'][0]]] = postags[child_dict['SBV'][0]]

        if 'VOB' in child_dict:
            triple['obj'][words[child_dict['VOB'][0]]] = postags[child_dict['VOB'][0]]

        if triple['sub'] and triple['obj']:
            return triple
        else:
            return {}


        # # 主谓宾
        # if 'SBV' in child_dict and 'VOB' in child_dict:
        #
        #     # e1 = words[child_dict['SBV'][0]]
        #     # e2 = words[child_dict['VOB'][0]]
        #     e1 = self.complete_e(words, postags, nertags, child_dict_list, child_dict['SBV'][0])
        #     e2 = self.complete_e(words, postags, nertags, child_dict_list, child_dict['VOB'][0])
        #     triple['sub'] = e1
        #     triple['obj'] = e2
        #     flag = True


        # svos.append([e1, r, e2])

        # # 定语后置，动宾关系
        # relation = format_parse_list[word_index][0] # 与父节点的关系
        # head = format_parse_list[word_index][2]  # ？ 不就是word_index吗
        # if relation == 'ATT':
        #     if 'VOB' in child_dict:
        #         e1 = self.complete_e(words, postags, nertags, child_dict_list, head - 1)
        #         r = words[word_index]
        #         e2 = self.complete_e(words, postags, nertags, child_dict_list, child_dict['VOB'][0])
        #         temp_string = r + e2
        #         if temp_string == e1[:len(temp_string)]:
        #             e1 = e1[len(temp_string):]
        #         if temp_string not in e1:
        #             svos.append([e1, r, e2])
        #
        #
        # # 含有介宾关系的主谓动补关系
        # # CMP 动补结构
        # if 'SBV' in child_dict and 'CMP' in child_dict:
        #     e1 = self.complete_e(words, postags, nertags, child_dict_list, child_dict['SBV'][0])
        #     cmp_index = child_dict['CMP'][0]
        #     r = words[word_index] + words[cmp_index]
        #     if 'POB' in child_dict_list[cmp_index]:
        #         e2 = self.complete_e(words, postags, nertags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
        #         svos.append([e1, r, e2])










    '''三元组抽取'''
    """
    遍历句子中每个词，作为verb词抽取
        1) 根据规则，根据 谓词的语义角色分析 抽取三元组（语义依存分析）
            主要依靠 SRL 来提取，因为毕竟自己构建的提取规则不够完善
            SRL提取中，会提取和 谓词 相关的 语义角色类型：
                语义角色类型：A0-动作的施事，A1-动作的影响，A2-A5，LOC-地点，MNR-方式，TMP-时间
        2) 若SRL抽取失败，根据规则，在 句法依存树 上抽取三元组（依存句法分析）
    """

    def extract(self, words, postags, nertags, child_dict_list, format_parse_list, roles_dict):
        svos = []

        for index in range(len(words)):
            word_pos = postags[index]

            if index in roles_dict:
                # 查看SRL提取结果
                # flag1, triple = self.according_srl_extract(words, postags, nertags, roles_dict, index)
                triple = self.according_srl_extract(words, postags, nertags, roles_dict, index)
                if (triple):
                    svos.append(triple)
            else:
                # 查看DP提取结果
                # flag2, triple = self.accroding_dp_extarct(words, postags, nertags, child_dict_list, format_parse_list, index)
                triple = self.accroding_dp_extarct(words, postags, nertags, child_dict_list, format_parse_list, index)
                if (triple):
                    svos.append(triple)

        return svos



    '''对找出的主语或者宾语进行扩展'''
    """
    如果不进行宾语补全的话会缺失很多信息
        例如：李克强总理今天来我家了,我感到非常荣幸。 对于谓词"来"，如果不补全的话，会得到 [总理, 来, 我家] 缺失了信息
        提出：*** 能不能在补全的时候加上NER标注信息，因为对于时间提取来说，主要提取的是实体之间的关系吧
    递归函数进行扩展，前缀和后缀扩展
        递归中对每个词当作同样对待，对其进行ATT扩展，以及如果是动词，进行VOB和SBV扩展
    """
    def complete_e(self, words, postags, nertags, child_dict_list, word_index):
        """
        对一个三元组中主语词和宾语词进行扩展，**由于主语/宾语可能是动词，动词又可能有主语和宾语，因此迭代的进行扩展
        :param words:
        :param postags:
        :param nertags:
        :param child_dict_list:
        :param word_index:
        :return:
        """
        child_dict = child_dict_list[word_index]

        # 前缀扩展
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, nertags, child_dict_list, child_dict['ATT'][i])


        # 后缀扩展
        postfix = ''
        if postags[word_index] == 'v': # 如果该词是动词才进行扩展
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, nertags, child_dict_list, child_dict['VOB'][0])
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, nertags, child_dict_list, child_dict['SBV'][0]) + prefix

        return prefix+ '\t' + words[word_index]+' '+postags[word_index]+' '+nertags[word_index] + postfix + '\t' #最终的都在这里*****（还是不是特别理解递归啊）






    '''文本三元组综合抽取主函数'''
    def triples_main(self, content):
        sentences = self.split_sents(content)
        svos = []
        for sentence in sentences:
            # 处理模型输出
            words, postags, nertags, child_dict_list, format_parse_list, roles_dict = self.parser.parser_main(sentence)
            # 三元组提取
            # svo = self.extract(words, postags, nertags, child_dict_list, format_parse_list, roles_dict)
            svo = self.extract(words, postags, nertags, child_dict_list, format_parse_list, roles_dict)
            print('\n\n-----------------------', sentence)
            print(svo)
            print(json.dumps(svo, indent=4, ensure_ascii=False))

            svos += svo

        return svos




if __name__ == '__main__':
    content1 = """环境很好，位置独立性很强，比较安静很切合店名，半闲居，偷得半日闲。点了比较经典的菜品，味道果然不错！烤乳鸽，超级赞赞赞，脆皮焦香，肉质细嫩，超好吃。艇仔粥料很足，香葱自己添加，很贴心。金钱肚味道不错，不过没有在广州吃的烂，牙口不好的慎点。凤爪很火候很好，推荐。最惊艳的是长寿菜，菜料十足，很新鲜，清淡又不乏味道，而且没有添加调料的味道，搭配的非常不错！"""
    content2 = """近日，一条男子高铁吃泡面被女乘客怒怼的视频引发热议。女子情绪激动，言辞激烈，大声斥责该乘客，称高铁上有规定不能吃泡面，质问其“有公德心吗”“没素质”。视频曝光后，该女子回应称，因自己的孩子对泡面过敏，曾跟这名男子沟通过，但对方执意不听，她才发泄不满，并称男子拍视频上传已侵犯了她的隐私权和名誉权，将采取法律手段。12306客服人员表示，高铁、动车上一般不卖泡面，但没有规定高铁、动车上不能吃泡面。
                高铁属于密封性较强的空间，每名乘客都有维护高铁内秩序，不破坏该空间内空气质量的义务。这也是乘客作为公民应当具备的基本品质。但是，在高铁没有明确禁止食用泡面等食物的背景下，以影响自己或孩子为由阻挠他人食用某种食品并厉声斥责，恐怕也超出了权利边界。当人们在公共场所活动时，不宜过分干涉他人权利，这样才能构建和谐美好的公共秩序。
                一般来说，个人的权利便是他人的义务，任何人不得随意侵犯他人权利，这是每个公民得以正常工作、生活的基本条件。如果权利可以被肆意侵犯而得不到救济，社会将无法运转，人们也没有幸福可言。如西谚所说，“你的权利止于我的鼻尖”，“你可以唱歌，但不能在午夜破坏我的美梦”。无论何种权利，其能够得以行使的前提是不影响他人正常生活，不违反公共利益和公序良俗。超越了这个边界，权利便不再为权利，也就不再受到保护。
                在“男子高铁吃泡面被怒怼”事件中，初一看，吃泡面男子可能侵犯公共场所秩序，被怒怼乃咎由自取，其实不尽然。虽然高铁属于封闭空间，但与禁止食用刺激性食品的地铁不同，高铁运营方虽然不建议食用泡面等刺激性食品，但并未作出禁止性规定。由此可见，即使食用泡面、榴莲、麻辣烫等食物可能产生刺激性味道，让他人不适，但是否食用该食品，依然取决于个人喜好，他人无权随意干涉乃至横加斥责。这也是此事件披露后，很多网友并未一边倒地批评食用泡面的男子，反而认为女乘客不该高声喧哗。
                现代社会，公民的义务一般分为法律义务和道德义务。如果某个行为被确定为法律义务，行为人必须遵守，一旦违反，无论是受害人抑或旁观群众，均有权制止、投诉、举报。违法者既会受到应有惩戒，也会受到道德谴责，积极制止者则属于应受鼓励的见义勇为。如果有人违反道德义务，则应受到道德和舆论谴责，并有可能被追究法律责任。如在公共场所随地吐痰、乱扔垃圾、脱掉鞋子、随意插队等。此时，如果行为人对他人的劝阻置之不理甚至行凶报复，无疑要受到严厉惩戒。
                当然，随着社会的发展，某些道德义务可能上升为法律义务。如之前，很多人对公共场所吸烟不以为然，烟民可以旁若无人地吞云吐雾。现在，要是还有人不识时务地在公共场所吸烟，必然将成为众矢之的。
                再回到“高铁吃泡面”事件，要是随着人们观念的更新，在高铁上不得吃泡面等可能产生刺激性气味的食物逐渐成为共识，或者上升到道德义务或法律义务。斥责、制止他人吃泡面将理直气壮，否则很难摆脱“矫情”，“将自我权利凌驾于他人权利之上”的嫌疑。
                在相关部门并未禁止在高铁上吃泡面的背景下，吃不吃泡面系个人权利或者个人私德，是不违反公共利益的个人正常生活的一部分。如果认为他人吃泡面让自己不适，最好是请求他人配合并加以感谢，而非站在道德制高点强制干预。只有每个人行使权利时不逾越边界，与他人沟通时好好说话，不过分自我地将幸福和舒适凌驾于他人之上，人与人之间才更趋于平等，公共生活才更趋向美好有序。"""
    content3 = '''（原标题：央视独家采访：陕西榆林产妇坠楼事件在场人员还原事情经过）
    央视新闻客户端11月24日消息，2017年8月31日晚，在陕西省榆林市第一医院绥德院区，产妇马茸茸在待产时，从医院五楼坠亡。事发后，医院方面表示，由于家属多次拒绝剖宫产，最终导致产妇难忍疼痛跳楼。但是产妇家属却声称，曾向医生多次提出剖宫产被拒绝。
    事情经过究竟如何，曾引起舆论纷纷，而随着时间的推移，更多的反思也留给了我们，只有解决了这起事件中暴露出的一些问题，比如患者的医疗选择权，人们对剖宫产和顺产的认识问题等，这样的悲剧才不会再次发生。央视记者找到了等待产妇的家属，主治医生，病区主任，以及当时的两位助产师，一位实习医生，希望通过他们的讲述，更准确地还原事情经过。
    产妇待产时坠亡，事件有何疑点。公安机关经过调查，排除他杀可能，初步认定马茸茸为跳楼自杀身亡。马茸茸为何会在医院待产期间跳楼身亡，这让所有人的目光都聚焦到了榆林第一医院，这家在当地人心目中数一数二的大医院。
    就这起事件来说，如何保障患者和家属的知情权，如何让患者和医生能够多一份实质化的沟通？这就需要与之相关的法律法规更加的细化、人性化并且充满温度。用这种温度来消除孕妇对未知的恐惧，来保障医患双方的权益，迎接新生儿平安健康地来到这个世界。'''
    content4 = '李克强总理今天来我家了,我感到非常荣幸'
    content5 = ''' 以色列国防军20日对加沙地带实施轰炸，造成3名巴勒斯坦武装人员死亡。此外，巴勒斯坦人与以色列士兵当天在加沙地带与以交界地区发生冲突，一名巴勒斯坦人被打死。当天的冲突还造成210名巴勒斯坦人受伤。
    当天，数千名巴勒斯坦人在加沙地带边境地区继续“回归大游行”抗议活动。部分示威者燃烧轮胎，并向以军投掷石块、燃烧瓶等，驻守边境的以军士兵向示威人群发射催泪瓦斯并开枪射击。'''
    content6 = '太原：16岁少年被五同学堵厕所围殴！向校园霸凌说不！（监控）'

    content7 = '李克强总理今天来我家了,我感到非常荣幸'
    extractor = TripleExtractor()

    svos = extractor.triples_main(content2)

    print('---------------', len(svos))



