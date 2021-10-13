from  model.SKIPGRAM import similarity_word100
from  model.SKIPGRAM import similarity_word128
from model.utils import load_parameter
import torch

def companyQuestionClassify(query,model, model2,vocab,program_section1, program_section2, program_section3):
    option = ['行业','公司','竞争','位置','总经理','法人','秘书','管理层','证券','经营']
    max_score,item_score = None,None
    cloest_option = ''
    
    for item in option:
        try:
            if max_score == None:
                max_score=similarity_word100(model,query,item)
                cloest_option = item
            else:
                item_score = similarity_word100(model,query,item)
                cloest_option = item if item_score>max_score else cloest_option
                max_score = item_score if item_score>max_score else max_score
        except KeyError:
            try:
                if max_score == None:
                    max_score=similarity_word128(model2,query,item,vocab,program_section1, program_section2, program_section3)
                    cloest_option = item
                else:
                    item_score = similarity_word128(model2,query,item,vocab,program_section1, program_section2, program_section3)
                    cloest_option = item if item_score>max_score else cloest_option
                    max_score = item_score if item_score>max_score else max_score
            except KeyError:
                cloest_option = '证券'
                break
    return cloest_option


def query_company(key,name,driver,model, model2,vocab,program_section1, program_section2, program_section3):
    option = companyQuestionClassify(key,model,model2,vocab,program_section1, program_section2, program_section3)
    msg_box, result = [], []
    if option == '行业':
        msg_box,result = driver.from_company_to_industry(name)
    
    if option == '公司' or option == '竞争':
        msg_box,result = driver.from_company_match_relative(name)
    elif option == '位置':
        msg_box,result = driver.from_company_to_position(name)
        
    elif option == '总经理':
        msg_box,result = driver.from_company_to_genmgr(name)
        
    elif option == '法人':
        msg_box,result = driver.from_company_to_legal(name)
        
    elif option == '秘书':
        msg_box,result = driver.from_company_to_Secbd(name)
        
    elif option == '管理层':
        msg_box,result = driver.from_company_to_allmgr(name)
        
    elif option == '证券':
        msg_box,result = driver.from_company_to_AffRepr(name)
        
    elif option == '经营':
        msg_box,result = driver.from_company_query_business(name)
        
    return msg_box, result


def askAssertClassify(word,skipgram,model):
	'''
	return 0 stands for ask
	return 1 stands for asert
	'''
	word_emb = torch.tensor(skipgram.word_vec(word))
	y_hat = model(word_emb)
	res = torch.argmax(y_hat).item()
	if res == 0:
		return 'ask'
	else:
		return 'asert'


def compIndClassify(word,skipgram,model):
    word_emb = torch.tensor(skipgram.word_vec(word))
    y_hat = model(word_emb)
    res = torch.argmax(y_hat).item()
    if res == 0:
        return 'company'
    else:
        return 'industry'

def get_personName(person_item):
    '''
    input: person_item
    return: person_name,person_position
    if name parameters don't exit, return None,None
    '''
    name_list = ['name_genmgr','name_legalRepr','name_AffRepr','name_Secbd']
    person_name, person_position = None, None
    for name in name_list:
        if dict(person_item).get(name) != None:
            person_name = dict(person_item).get(name)
            person_position = name[5:]
            break
    return person_name,person_position

def eng2cn(eng):
    if eng == 'genmgr':
        return '总经理'
    elif eng == 'legalRepr':
        return '法人代表'
    elif eng == 'AffRepr':
        return '证券事务代表'
    elif eng == 'Secbd':
        return '董事会秘书'
    elif eng == 'ask':
        return '询问'
    elif eng == 'asert':
        return '断言'
    if eng == 'company':
        return '同地点公司查询'
    if eng == 'industry':
        return '同地点同行业公司查询'
    else:
        return eng

def printList(arr):
    total = ''
    for item in arr:
        item_enter = item + '\n'
        total += item_enter
        print(item)
    return total


def LCstring(string1,string2):
    '''
    return record matrix, LCstring of <String1,String2>
    for example:
        LCS("helloworld","loop")
        [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3],
        [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
        ], 3
    '''
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    result = 0
    pos = 0
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
                if res[i][j] > result:
                    pos = i-1
                    result = res[i][j]

    return result,pos-result+1


def LCS(string1,string2):
    '''
    return record matrix, LCS of <String1,String2>
    for example:
        LCString("helloworld","loop")
        2" for has common substring 'lo'
    '''
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
            else:
                res[i][j] = max(res[i-1][j],res[i][j-1])
    return res,res[-1][-1]


def edit_distance(str1, str2):
    '''
        return minimal edit distance of str1 and str2
    '''
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1,len(str1)+1):
        for j in range(1,len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
    return matrix[len(str1)][len(str2)]

def closet_element_LCS(query,total):
    max_length = 0
    closet_item = ''
    closet_item_list_LCS = []
    for item in total:
        _, lcs_tmp = LCS(item, query)
        if(lcs_tmp>max_length):
            closet_item_list_LCS = []
            max_length = lcs_tmp
        if(lcs_tmp==max_length):
            closet_item_list_LCS.append(item)

    substring_length = 0
    for item in closet_item_list_LCS:
        substring_length, pos= LCstring(query,item)
        if pos == 0 or pos+substring_length == len(query):
            return item,max_length

    min_edit_length = 1000
    for item in closet_item_list_LCS:
        lcstr_tmp = edit_distance(item,query)
        if(lcstr_tmp<min_edit_length):
            closet_item = item
            min_edit_length = lcstr_tmp

    return closet_item, max_length

def status_correctness(sentence):
    sentence = sentence.replace('董事会秘书','秘书',1)
    sentence = sentence.replace('代表','',1)
    sentence = sentence.replace('互联网','信息',1)
    sentence = sentence.replace('科技行业','信息',1)
    sentence = sentence.replace('科技业','信息',1)
    sentence = sentence.replace('科创业','信息',1)
    sentence = sentence.replace('科创','信息',1)
    sentence = sentence.replace('法人代表','法人',1)
    sentence = sentence.replace('管理人员','职务',1)
    sentence = sentence.replace('事务师','',1)
    sentence = sentence.replace('事务','',1)    
    return sentence

def blur_correctness(seg_word,lex_word,total_person,total_province,total_company):
    '''
    return buffer, person_valid, company_valid, province_valid
    buffer: sentence through blur correctness
    person_valid,company_valid,province_valid: 0 means invalid, 1 means totally invalid, 2 means modify
    '''
    person_valid = 1
    province_valid = 1
    company_valid = 1
    sentence_valid = 0
    buffer = []
    for idx, word in enumerate(seg_word):
        LCS_length = 0
        bufword = word
        if lex_word[idx] == 'PER':
            sentence_valid += 1
            closet_item, max_length = closet_element_LCS(word,total_person)
            if max_length <= 1:
                person_valid = 0
            else:
                if closet_item == seg_word[idx]:
                    person_valid = 1
                else:
                    person_valid = 2
                bufword = closet_item
    
        if lex_word[idx] == 'LOC':
            sentence_valid += 1
            closet_item, max_length = closet_element_LCS(word,total_province)
            if max_length >= 2:
                if closet_item == seg_word[idx]:
                    province_valid = 1
                else:
                    province_valid = 2
                bufword = closet_item
            else:
                province_valid = 0
        
        if lex_word[idx] == 'ORG' or lex_word[idx] == 'nt':
            sentence_valid += 1
            closet_item, max_length = closet_element_LCS(word,total_company)
            if max_length > len(word)//3:
                if closet_item == seg_word[idx]:
                    company_valid = 1
                else:
                    company_valid = 2
                bufword = closet_item
            else:
                company_valid = 0
        buffer.append(bufword)
    if sentence_valid == 0:
        person_valid, company_valid, province_valid = sentence_valid, sentence_valid, sentence_valid

    return buffer, person_valid, company_valid, province_valid

def existence_check(seg_lex):
    '''
    return loc_exit, per_exit, cmp_exit
    '''
    loc_exit = 1 if 'LOC' in seg_lex else 0
    per_exit = 1 if 'PER' in seg_lex else 0
    cmp_exit = 1 if 'ORG' in seg_lex or 'nt' in seg_lex else 0
    return loc_exit, per_exit, cmp_exit

def industry_existence(query,total_industry):
    '''
    return closet_item, max_length
    '''
    closet_item, max_length = closet_element_LCS(query,total_industry)
    return closet_item, max_length





if __name__ == '__main__':
    '''
    PATH_PERSON = '.\src\model\ckpts\person_name.pkl'
    PATH_PROVINCE = '.\src\model\ckpts\province_name.pkl'
    PATH_COMPANY = '.\src\model\ckpts\company_name.pkl'
    total_person = load_parameter(PATH_PERSON)
    total_province = load_parameter(PATH_PROVINCE)
    total_company = load_parameter(PATH_COMPANY)
    '''
    PATH_INDUSTRY = '.\src\model\ckpts\industry.pkl'
    total_industry = load_parameter(PATH_INDUSTRY)



    print(industry_existence('科技',total_industry))

    '''
    seg_word = ['告诉我', '杨予光', '在', '金地', '担任', '什么', '职务']
    seg_lex = ['v', 'PER', 'p', 'ORG', 'v', 'r', 'n']

    buf, person_valid, company_valid, province_valid = blur_correctness(seg_word, seg_lex, total_person, total_province, total_company)
    print(buf)
    print(person_valid)
    print(company_valid)

    print(closet_element_LCS('珠海华发',['青海华鼎','华发股份','上海华青']))
    '''