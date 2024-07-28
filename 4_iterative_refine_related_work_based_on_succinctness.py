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
import nltk


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

@backoff.on_exception(backoff.expo, RateLimitError)
def generator(message):
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0125",
        # model="gpt-4-turbo-preview",
        model="gpt-4-0125-preview",
        messages = message,
        temperature=0,
        max_tokens= 2500,
    )
    ini_sent = response.choices[0]['message']['content']
    # response = completion_with_backoff(model="gpt-3.5-turbo-instruct", prompt=d, max_tokens=1500, temperature=0)
    # ini_sent = response["choices"][0]["text"]
    # import pdb; pdb.set_trace()
    # ini_sent = response.choices[0]['message']['content']

    return ini_sent    

# path_prompt = r'/root/wpc/Work5/prompts/prompt2_related_work_generation_0shot.txt')
path_prompt = r'/root/wpc/Work5_2/prompts_new/prompt_iiterative_check_step4_generate_related_work_basedon_Succinctness.txt'
with open(path_prompt,'r',encoding='utf-8') as f:
    prompt = f.read()

system_message = {"role": "system", "content": prompt}

path_api = r'/root/wpc/Work5/API_KEYS_42'
api_list = get_api_keys(path_api)

path_temp = r'/root/wpc/Work5/temp.txt'


path_input_text = r'/root/wpc/Work5_2/dataset/complete_related_work_dataset30.json'
path_input_meta = r'/root/wpc/Work5_2/dataset/meta_element30.json'
path_input_0sum = r'/root/wpc/Work5_2/result_new/gpt4_meta_iterative_related_work_basedon_structure_clarity_order_total3.txt'
path_input_eval = r'/root/wpc/Work5_2/result_new/gpt4_related_work_succinctness_evaluation2.json'

path_w_temp = r'/root/wpc/Work5/result_new/gpt35_meta_element_contribution_limitation.json'

path_w = r'/root/wpc/Work5_2/result_new/gpt4_meta_iterative_related_work_basedon_succinctness22.json'
results = []
with open(path_input_text,'r',encoding='utf-8') as g_text, open(path_input_meta,'r',encoding='utf-8') as g_meta, \
    open(path_input_0sum,'r',encoding='utf-8') as g_0sum, open(path_input_eval,'r',encoding='utf-8') as g_eval,  open(path_w,'a',encoding='utf-8') as h:
    lines_text = g_text.readlines()
    lines_meta = g_meta.readlines()
    lines_0sum = g_0sum.readlines()
    lines_eval = g_eval.readlines()

    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    api_index = 0

    start = 16
    end = 30
    line_index = [2,13,16,17,19]
    # for line_text, line_meta, line_0sum, line_eval in zip(lines_text[start:end],lines_meta[start:end], lines_0sum[start:end], lines_eval[start:end]):
    for line_i in line_index:
        line_text = lines_text[line_i-1]
        line_meta = lines_meta[line_i-1]
        line_0sum = lines_0sum[line_i-1]
        line_eval = lines_eval[line_i-1]

        openai.api_key = api_list[api_index]

        message = []
        line = json.loads(line_text)

        line_meta = json.loads(line_meta)

        dic = {}

        dic['Target Paper'] = {}
        dic['Target Paper']['title'] = line['target_paper']['title']
        dic['Target Paper']['abstract'] = line['target_paper']['abstract']
        dic['Target Paper']['introduction'] = line['target_paper']['introduction']
        dic['Target Paper']['conclusion'] = line['target_paper']['conclusion']

        del line_meta['id']

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

        dic['Related Work Draft'] = {}

        sent = nltk.sent_tokenize(line_0sum.strip())

        for index,s in enumerate(sent):
            dic["Related Work Draft"]["<SENTENCE_"+str(index+1)+">"] = s

        # dic['Feedback from the Review'] = line_eval

        line_eval = json.loads(line_eval)

        dic['Feedback from the Reviewer'] = line_eval['Succinctness Problem']

        # pdb.set_trace()
        with open(path_temp,'w',encoding='utf-8') as f:
            f.write(json.dumps(dic, ensure_ascii=False, indent=1))
            f.write('\n')

        with open(path_temp, 'r', encoding='utf-8') as f:

            inputs = f.read()

            user_prompt = 'Now let us begin, I will give you the input:\nInput:\n' + inputs

            user_prompt += '\nNow please generate the output in the required JSON format:\nOutput:\n'

            user_message = {"role": "user", "content":user_prompt}
            message.append(system_message) 
            message.append(user_message) 

            total_count = len(tokenizer.encode(prompt+user_prompt))
            # print(total_count)
            output =generator(message)
            # pdb.set_trace()

            output = output.replace('\n',' ').replace("```json",'').replace("```",'')

            h.write(output)
            h.write('\n')
            
            h.flush()

            if api_index == len(api_list)-1:
                api_index = 0
            else:
                api_index += 1

        # time.sleep(15)
