from flask import Flask,request
from flask import render_template
import paddlehub as hub
import QAsystem as QA
import torch
from os.path import abspath
from model.BiGRU import analysis
from model.mlp import MLP
from model.SKIPGRAM import top_semantic128_noun
from model.utils import load_parameter
from gensim.models import KeyedVectors
from paddlehub.reader.tokenization import load_vocab
import paddle.fluid as fluid
from utils import companyQuestionClassify,askAssertClassify,blur_correctness,status_correctness,industry_existence,compIndClassify
import utils
from Neo4jDriver import Neo4jDriver



app = Flask(__name__)
print('start loading MLP model')
mlp_model = MLP()
mlp_model.load_state_dict(torch.load('.\model\ckpts\mlp.pt'))
mlp2_model = MLP()
mlp2_model.load_state_dict(torch.load('.\model\ckpts\mlp2.pt'))
    
print('start loading Bi-GRU model')
lac = hub.Module(name='lac')
print('start loading skip-gram model, <vector:100dim>')
skipgram100 = KeyedVectors.load_word2vec_format(abspath('.')+"\model\word_model\skipgram.bin", \
    binary = True, encoding = "utf-8", unicode_errors = "ignore")

print('start loading skip-gram model, paddlehub, <vector:128dim>')
skipgram128 = hub.Module(name="word2vec_skipgram")
vocab = load_vocab(skipgram128.get_vocab_path())
inputs, outputs, program = skipgram128.context(trainable=False,max_seq_len=1)

fluid.disable_dygraph()
word_ids = inputs["text"]
embedding = outputs["emb"]

place = fluid.CPUPlace()  
exe = fluid.Executor(place)  
feeder = fluid.DataFeeder(feed_list=[word_ids], place=place)

sec1 = (inputs, outputs, program)
sec2 = (word_ids, embedding)
sec3 = (place, exe, feeder)

print('start loading all entity name into list below')
PATH_PERSON = '.\model\ckpts\person_name.pkl'
PATH_PROVINCE = '.\model\ckpts\province_name.pkl'
PATH_COMPANY = '.\model\ckpts\company_name.pkl'
PATH_INDUSTRY = '.\model\ckpts\industry.pkl'
total_person = load_parameter(PATH_PERSON)
total_province = load_parameter(PATH_PROVINCE)
total_company = load_parameter(PATH_COMPANY)
total_industry = load_parameter(PATH_INDUSTRY)

print('start connecting to neo4j console')
bolt_url = "bolt://localhost:7687"
user = "neo4j"
password = "Neo4j7474$"
Neo4jDriver = Neo4jDriver(bolt_url, user, password)

def SemanticQuery(question):
    msg_res = ''
    question = status_correctness(question)
    
    seg_word, seg_lex = analysis(lac,question)
    seg_word,seg_lex = seg_word[0],seg_lex[0]
    buffer, person_valid, company_valid, province_valid = blur_correctness(seg_word,seg_lex,total_person,total_province,total_company)
    seg_word = buffer
    if len(seg_word) == 1:
        print('分词结果:',seg_word)
        print('词性标注:',seg_lex)
        if seg_lex[0] == 'LOC' and province_valid !=0:
            msg_box, result = Neo4jDriver.from_location_to_company(seg_word[0])
        elif seg_lex[0] == 'PER' and person_valid !=0:
            msg_box, result = Neo4jDriver.from_person_to_company(seg_word[0])
        elif seg_lex[0] == 'ORG' and company_valid!=0:
            msg_box, result = Neo4jDriver.from_company_query_business(seg_word[0])
        else:
            msg_box, result = [], []
            if seg_lex[0] == 'LOC' and province_valid ==0:
                msg_res = '目前只支持在省级范围内搜索公司\n'
            elif seg_lex[0] == 'PER' and person_valid == 0:
                msg_res = '人名是无效的,要不换个试试？'
            elif seg_lex[0] == 'ORG' and company_valid == 0:
                msg_res = '公司名是无效的,要不换个试试?'
            else:
                msg_res = '命名实体识别系统失效, 请输入完整的句子\n'
        msg_res += utils.printList(msg_box)
        return msg_res

    elif len(seg_word) == 0:
        msg_res = '分词系统失效, 请输入完整的句子\n'
        return msg_res

    print('分词结果: ',seg_word)
    print('实体标注: ',seg_lex)
    loc_exit, per_exit, cmp_exit = utils.existence_check(seg_lex)
    if loc_exit == 1 and cmp_exit == 0:
        if province_valid == 0:
            msg_res = '目前只支持在省级范围内搜索公司\n' 
            return msg_res
        company_temp,company_lex_temp,ind_flg = [], [],0
        key = ''
        for idx, lex in enumerate(seg_lex):
            if lex == 'n' or lex == 'vn':
                question = compIndClassify(seg_word[idx],skipgram100,mlp2_model)
                if question == 'industry':
                    key = seg_word[idx]
                    ind_flg = 1
                    break
                if question == 'company':
                    key = seg_word[idx]
                    company_temp.append(key)
                    company_lex_temp.append(lex)
                    continue
        if ind_flg == 0:
            key = top_semantic128_noun(skipgram128,company_temp,company_lex_temp,vocab,sec1,sec2,sec3)
        location_name = seg_word[seg_lex.index('LOC')]

        print('关键名词抽取:',key)
        if question == 'company':
            seg_box, result = Neo4jDriver.from_location_to_company(location_name)
        else:
            closet_item, max_length = industry_existence(key, total_industry)
            if max_length >= 2:
                if closet_item != key:
                    msg_res += '行业名在数据库中未找到,根据相似度算法替换为{}\n'.format(closet_item)
                seg_box, result = Neo4jDriver.from_location_match_industry_for_company(location_name,closet_item)
                if len(seg_box) == 0:
                    msg_res += '{location}没有{industry}行业的公司\n'.format(location=location_name,industry=key)
            else:
                msg_res += '行业名在数据库中未找到,且无相似行业,输出该省所有公司\n'
                seg_box, result = Neo4jDriver.from_location_to_company(location_name)

        msg_res += utils.printList(seg_box)
        return msg_res

    if per_exit == 0 and cmp_exit == 1:
        if company_valid == 0:
            msg_res += '公司名在数据库中未找到,且无相似公司,要不换个试试？\n'
            return msg_res
        if company_valid == 2:
            try:
                msg_res += '公司名在数据库中未找到,根据相似度算法替换为{name}\n'.format(name=seg_word[seg_lex.index('ORG')])
            except:
                msg_res += '公司名在数据库中未找到,根据相似度算法替换为{name}\n'.format(name=seg_word[seg_lex.index('nt')])
        if company_valid != 0:
            try:
                key = top_semantic128_noun(skipgram128,seg_word,seg_lex,vocab,sec1,sec2,sec3)
            except KeyError:
                msg_res += '这..这个词要么是错别字要么就太生僻了\n'
                return msg_res
            print('关键名词抽取:',key)
            if key == None:
                msg_res += '关键词系统故障,请输入完整的句子\n'
                return msg_res
            print('关键名词抽取:',key)
            tmp_lex = []
            company_name = ''
            seg_word_tmp = seg_word.copy()
            for idx, lex in enumerate(seg_lex):
                if (lex == 'n' and seg_word_tmp[idx] != key) or (lex == 'nz' and seg_word_tmp[idx] != key) or  (lex == 'vn' and seg_word_tmp[idx] != key):
                    rm_word = seg_word_tmp[idx]
                    seg_word.remove(rm_word)
                    continue
                if lex == 'nt' or lex == 'ORG':
                    company_name = seg_word_tmp[idx]
                tmp_lex.append(lex)
            seg_lex = tmp_lex
            query_word = key
            msg_box, result = utils.query_company(query_word,company_name,Neo4jDriver,skipgram100,skipgram128,vocab,sec1,sec2,sec3)
            msg_res += utils.printList(msg_box)
            return msg_res
    
    if per_exit == 1 and cmp_exit == 0:
        if person_valid == 0:
            msg_res += '不论是人名是无效的,要不换个试试？\n'
            return msg_res
        if person_valid == 2:
            msg_res += '人名在数据库中未找到,根据相似度算法替换为{name}\n'.format(name=seg_word[seg_lex.index('PER')])
        if person_valid != 0:
            person_name = seg_word[seg_lex.index('PER')]
            msg_box, result = Neo4jDriver.from_person_to_company(person_name)
            msg_res += utils.printList(msg_box)
            return msg_res

    if per_exit == 1 and cmp_exit == 1:
        if person_valid == 0 and company_valid == 0:
            msg_res += '不论是人名还是公司名都是无效的,要不换个试试？\n'
            return msg_res
        if person_valid == 2:
            msg_res += '人名在数据库中未找到,根据相似度算法替换为{name}\n'.format(name=seg_word[seg_lex.index('PER')])
        if company_valid == 2:
            try:
                msg_res += '公司名在数据库中未找到,根据相似度算法替换为{name}\n'.format(name=seg_word[seg_lex.index('ORG')])
            except ValueError:
                msg_res += '公司名在数据库中未找到,根据相似度算法替换为{name}\n'.format(name=seg_word[seg_lex.index('nt')])
        if person_valid == 0 and company_valid!=0:
            '''
            situation1: 问句, <PERSON>-<company>
            situation2: 断言, <PERSON>-<company>
            '''
            msg_res += '人名不存在,将根据句子中的剩余信息进行意图推断\n'
            key = top_semantic128_noun(skipgram128,seg_word,seg_lex,vocab,sec1,sec2,sec3)
            tmp_lex = []
            seg_word_tmp = seg_word.copy()
            company_name = ''
            for idx, lex in enumerate(seg_lex):
                if (lex == 'n' and seg_word_tmp[idx] != key) or (lex == 'nz' and seg_word_tmp[idx] != key) or  (lex == 'vn' and seg_word_tmp[idx] != key):
                    rm_word = seg_word_tmp[idx]
                    seg_word.remove(rm_word)
                    continue
                if lex == 'nt' or lex == 'ORG':
                    company_name = seg_word_tmp[idx]
                tmp_lex.append(lex)
            seg_lex = tmp_lex
            flag3 = seg_lex[seg_word.index(key)]
            print('关键名词抽取:',key)
            try:
                person, tone = QA.tri_inference(seg_word,seg_lex,flag1='PER',flag2='ORG',flag3=flag3,model=(skipgram100,mlp_model))
            except ValueError:
                person, tone = QA.tri_inference(seg_word,seg_lex,flag1='PER',flag2='nt',flag3=flag3,model=(skipgram100,mlp_model))
            if tone == 'ask':
                msg_box, result = Neo4jDriver.from_company_to_allmgr(company_name)
                msg_res += utils.printList(msg_box)
                return msg_res
            if tone == 'asert':
                msg_box, result = utils.query_company(key, company_name, Neo4jDriver, skipgram100, skipgram128, vocab, sec1, sec2, sec3)
                msg_res += utils.printList(msg_box)
                return msg_res
            
        if person_valid != 0 and company_valid == 0:
            '''
                situation1: 问句, <PERSON>-<company>
                situation2: 断言, <PERSON>-<company>
            '''
            msg = '公司名不存在,将根据句子中的剩余信息进行意图推断\n'
            msg_res += msg
            key = top_semantic128_noun(skipgram128,seg_word,seg_lex,vocab,sec1,sec2,sec3)
            tmp_lex = []
            person_name = ''
            for idx, lex in enumerate(seg_lex):
                if lex == 'PER':
                    person_name = seg_word[idx]
                    break
                
            seg_lex = tmp_lex
            msg_box, result = Neo4jDriver.from_person_to_company(person_name)
            msg_res += utils.printList(msg_box)
            return msg_res
        

        if person_valid != 0 and company_valid!=0:
            '''
            situation1: 问句, <PERSON>-<company>
            situation2: 断言, <PERSON>-<company>
            '''
            key = top_semantic128_noun(skipgram128,seg_word,seg_lex,vocab,sec1,sec2,sec3)
            print('关键名词抽取:',key)
            tmp_lex = []
            company_name = ''
            seg_word_tmp = seg_word.copy()
            for idx, lex in enumerate(seg_lex):
                if (lex == 'n' and seg_word_tmp[idx] != key) or (lex == 'nz' and seg_word_tmp[idx] != key) or  (lex == 'vn' and seg_word_tmp[idx] != key):
                    rm_word = seg_word_tmp[idx]
                    seg_word.remove(rm_word)
                    continue
                if lex == 'nt' or lex == 'ORG':
                    company_name = seg_word_tmp[idx]
                tmp_lex.append(lex)
            seg_lex = tmp_lex
            flag3 = seg_lex[seg_word.index(key)]
            try:
                person, tone = QA.tri_inference(seg_word,seg_lex,flag1='PER',flag2='ORG',flag3=flag3,model=(skipgram100,mlp_model))
            except ValueError:
                person, tone = QA.tri_inference(seg_word,seg_lex,flag1='PER',flag2='nt',flag3=flag3,model=(skipgram100,mlp_model))
            if tone == 'ask':
                msg_box, result = Neo4jDriver.from_company_to_allmgr(company_name)
                msg_res += utils.printList(msg_box)
                return msg_res
            if tone == 'asert':
                msg_box, result = utils.query_company(key, company_name, Neo4jDriver, skipgram100, skipgram128, vocab, sec1, sec2, sec3)
                msg_res += utils.printList(msg_box)
                return msg_res

@app.route('/')
def reader():
    return render_template('KG1.html',output="zou")

@app.route('/solve')    
def solve():
    params = request.args.get('params')
    # return render_template('reader.html',output="zou")

    
    if SemanticQuery(params) != None:
        QueryResult = SemanticQuery(params).split("\n")
    else:
        QueryResult = ['啊这...我是查询系统不是聊天机器人']
        
    return render_template('KG1.html',output=QueryResult)

if __name__ == '__main__':

    flag = 1
    while(flag):
        question = input('neo4j ask:')
        print(SemanticQuery(question))
    
    Neo4jDriver.close()