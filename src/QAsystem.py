import paddlehub as hub
import torch
from os.path import abspath
from model.BiGRU import analysis
from model.mlp import MLP
from model.utils import load_parameter
from gensim.models import KeyedVectors
from paddlehub.reader.tokenization import load_vocab
from utils import companyQuestionClassify,askAssertClassify,blur_correctness,compIndClassify


def single_inference(seg_word,seg_lex,flag):
    '''
    single means  there is only one valid entity(PER,LOC,ORG) in seg_lex
    '''
    flag_index = seg_lex.index(flag)
    return seg_word[flag_index]


def bi_inference(seg_word,seg_lex,flag1,flag2,model):
    '''
    model = (skipgram100, mlp2)
    '''
    flag1_index = seg_lex.index(flag1)
    flag2_index = seg_lex.index(flag2)
    skipgram, mlp2 = model
    '''
    事实性问句：xxx公司的yyy？
    '''
    if flag1 == 'ORG' or flag1 == 'nt':
        query_word = seg_word[flag2_index]
        return query_word

    if flag1 == 'LOC' and flag2 == 'n':
        query_word = seg_word[flag1_index]
        try:
            key = compIndClassify(seg_word[flag2_index],skipgram,mlp2)
        except KeyError:
            key = 'industry'
        return query_word,key
    
def tri_inference(seg_word,seg_lex,flag1,flag2,flag3,model,vocab=None):
    '''
    model = (skipgram100, skipgram150, mlp)
    '''
    skipgram100, mlp_model = model
    flag1_index = seg_lex.index(flag1)
    flag2_index = seg_lex.index(flag2)
    flag3_index = seg_lex.index(flag3)

    if flag3 == 'n' or flag3 == 'nz':
        tone = 'ask'
        flag3_word = seg_word[flag3_index]
        tone_class = askAssertClassify(flag3_word,skipgram100,mlp_model)
        tone = 'ask' if tone_class=='ask' else 'asert'
             
        if (flag1,flag2) == ('ORG','PER') or (flag1,flag2) == ('nt','PER'): 
            return seg_word[flag2_index],tone
        if (flag1,flag2) == ('PER','ORG') or (flag1,flag2) == ('PER','nt'):
            return seg_word[flag1_index],tone


if __name__ == '__main__':

    print('start loading Bi-GRU model')
    lac = hub.Module(name='lac')
    seg_word, seg_lex = analysis(lac,'与长春电信有关的公司')
    print(seg_word,seg_lex)

    '''
    print('start loading skip-gram model, <vector:100dim>')
    skipgram100 = KeyedVectors.load_word2vec_format(abspath('.')+"\src\model\word_model\skipgram.bin", \
			binary = True, encoding = "utf-8", unicode_errors = "ignore")

    mlp2_model = MLP()
    PATH = '.\src\model\ckpts\mlp2.pt'
    mlp2_model.load_state_dict(torch.load(PATH))



    print(bi_inference(['广东','有','公司'],['LOC','v','n'],'LOC','n',(skipgram100,mlp2_model)))
    print('start loading skip-gram model, paddlehub, <vector:150dim>')
    skipgram150 = hub.Module(name="word2vec_skipgram")
    vocab = load_vocab(skipgram150.get_vocab_path())

    print('start loading all entity name into list below')
    PATH_PERSON = '.\src\model\ckpts\person_name.pkl'
    PATH_PROVINCE = '.\src\model\ckpts\province_name.pkl'
    PATH_COMPANY = '.\src\model\ckpts\company_name.pkl'
    total_person = load_parameter(PATH_PERSON)
    total_province = load_parameter(PATH_PROVINCE)
    total_company = load_parameter(PATH_COMPANY)

    blur_correctness(seg_word,seg_lex,total_person,total_province,total_company)

    print('start loading MLP model')
    mlp_model = MLP()
    PATH = '.\src\model\ckpts\mlp.pt'
    mlp_model.load_state_dict(torch.load(PATH))
    print(askAssertClassify('文案',skipgram100,mlp_model))
    
    print(companyQuestionClassify('财务',skipgram100,model2=skipgram150,vocab=vocab))
    '''

    

    