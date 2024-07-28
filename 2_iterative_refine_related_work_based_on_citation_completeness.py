import backoff
import json
import openai
from openai.error import RateLimitError
import tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  
import pdb
import time
import tiktoken
from tqdm import tqdm
import re
import string


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def get_api_keys(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        api_lists = []
        for line in lines:
            api = line.strip().split('----')[-1]
            api_lists.append(api)
    return api_lists

def count_text(dic,tokenizer, prompt):
    all_content = []
    for key in dic['Target Paper']:
        all_content.append(dic['Target Paper'][key])
    for key in dic['Reference Papers']:
        for k in dic['Reference Papers'][key]:
            all_content.append(dic['Reference Papers'][key][k])
    all_content = ' '.join(all_content)
    all_content += prompt
    text_list = tokenizer.encode(all_content)
    
    return len(text_list)

def count_cite(text):
    text = text.replace('\\"','"')
    t = re.findall(r'@cite_\d{1,2}',text)
    t = set(t)

    return len(t)

def interpret(output):
    out_dic = {}
    start = output.find("Refined Related Work")
    stop = output.find("Modification Operations")

    rw = output[start+len("Refined Related Work"):stop-1]
    rw = rw.strip().strip(string.punctuation).strip()

    out_dic['Refined Related Work'] = rw

    modi = output[stop+len("Modification Operations")+2:-2]

    out_dic['Modification Operations'] = modi

    return out_dic

@backoff.on_exception(backoff.expo, RateLimitError)
def generator(message):
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0125",
        # model="gpt-4-turbo-preview",
        model="gpt-4-0125-preview",
        messages = message,
        temperature=0,
        max_tokens= 1800,
    )
    ini_sent = response.choices[0]['message']['content']
    # response = completion_with_backoff(model="gpt-3.5-turbo-instruct", prompt=d, max_tokens=1500, temperature=0)
    # ini_sent = response["choices"][0]["text"]
    # import pdb; pdb.set_trace()
    # ini_sent = response.choices[0]['message']['content']

    return ini_sent    

# path_prompt = r'/root/wpc/Work5/prompts/prompt2_related_work_generation_0shot.txt')
path_prompt = r'/root/wpc/Work5_2/prompts_new/prompt_iiterative_check_step1_evaluation_citation_completeness.txt'
with open(path_prompt,'r',encoding='utf-8') as f:
    prompt = f.read()

system_message = {"role": "system", "content": prompt}

path_api = r'/root/wpc/Work5/API_KEYS_4'
api_list = get_api_keys(path_api)

path_temp = r'/root/wpc/Work5/temp.txt'


path_input_text = r'/root/wpc/Work5_2/dataset/complete_related_work_dataset30.json'
path_input_meta = r'/root/wpc/Work5_2/dataset/meta_element30.json'
path_input_init = r'/root/wpc/Work5_2/result_new/gpt4_complete_abstract_0shot30.json'

path_w_temp = r'/root/wpc/Work5/result_new/gpt35_meta_element_contribution_limitation.json'
path_w_temp_inloop = r'/root/wpc/Work5_2/result_new/temp_inloop.json'
path_w_temp_outloop = r'/root/wpc/Work5_2/result_new/temp_outloop.json'

path_w = r'/root/wpc/Work5_2/result_new/gpt4_iterative_related_work_citation_complete_total.json'
results = []
with open(path_input_text,'r',encoding='utf-8') as g_text, open(path_input_meta,'r',encoding='utf-8') as g_meta, \
    open(path_input_init,'r',encoding='utf-8') as g_init:
    lines_text = g_text.readlines()
    lines_meta = g_meta.readlines()
    lines_init = g_init.readlines()

    api_index = 1

    out_f = open(path_w_temp_outloop,'a',encoding='utf-8')

    for line_text, line_meta, line_init in zip(lines_text[10:],lines_meta[10:], lines_init[10:]):
        openai.api_key = api_list[api_index]

        line = json.loads(line_text)

        line_meta = json.loads(line_meta)

        gold = line['related_work']

        count_gold_cite = count_cite(gold)

        now_draft = line_init.strip()
        pre_draft = ''

        total_result = {}
        del line_meta['id']
        for loop_step in range(3):

            recall_cite_pre = count_cite(pre_draft)
            recall_cite_now = count_cite(now_draft)

            if count_gold_cite == recall_cite_now or recall_cite_pre == recall_cite_now or recall_cite_pre > recall_cite_now:
                break

            dic = {}
            message = []

            dic['Target Paper'] = {}
            dic['Target Paper']['title'] = line['target_paper']['title']
            dic['Target Paper']['abstract'] = line['target_paper']['abstract']
            dic['Target Paper']['introduction'] = line['target_paper']['introduction']
            dic['Target Paper']['conclusion'] = line['target_paper']['conclusion']

            references = line_meta

            # pdb.set_trace()

            dic_write = {}

            dic_write['id'] = line['id']

            dic['Reference Papers'] = {}

            dic['Reference Papers']['Total citation identifiers'] = []
            preserve_keys = []
            for ref in references:
                refnew = ref.strip()
                dic['Reference Papers']['Total citation identifiers'].append(refnew)
                dic['Reference Papers'][refnew] = {}

                for key in references[ref]:
                    dic['Reference Papers'][refnew][key] = references[ref][key]

                preserve_keys.append(refnew)

            dic['Draft Related Work'] = now_draft

            pre_draft = now_draft

            with open(path_temp,'w',encoding='utf-8') as f:
                f.write(json.dumps(dic, ensure_ascii=False, indent=1))
                f.write('\n')

            with open(path_temp, 'r', encoding='utf-8') as f:

                inputs = f.read()

                user_prompt = 'Now let us begin, I will give you the input:\n' + inputs

                user_prompt += '\nNow please think step by step according to the required steps. Then please output only the refined related work as well as your modification actions in the required JSON format:'

                user_message = {"role": "user", "content":user_prompt}
                message.append(system_message) 
                message.append(user_message) 

                # pdb.set_trace()

                output =generator(message)

                output = output.replace('\n',' ')

                output = output.replace(r"```json", '').replace(r"```",'').strip()

                output_dict = interpret(output)

                refined_related_work = output_dict["Refined Related Work"]
                total_result["step "+str(loop_step+1)] = output_dict
                now_draft = refined_related_work

                out_f.write(json.dumps(output_dict, ensure_ascii=False, indent=1))
                out_f.write('\n')
                out_f.flush()

                if api_index == len(api_list)-1:
                    api_index = 0
                else:
                    api_index += 1

        with open(path_w,'a',encoding='utf-8') as tt:
            tt.write(json.dumps(total_result))
            tt.write('\n')

            tt.flush()