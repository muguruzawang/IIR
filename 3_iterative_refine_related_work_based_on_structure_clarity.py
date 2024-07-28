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

    return ini_sent    

def evaluator(message):
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0125",
        # model="gpt-4-turbo-preview",
        model="gpt-4-0125-preview",
        messages = message,
        temperature=0.7,
        max_tokens= 2500,
    )
    ini_sent = response.choices[0]['message']['content']

    return ini_sent

# path_prompt = r'/root/wpc/Work5/prompts/prompt2_related_work_generation_0shot.txt')
path_prompt_eval = r'/root/wpc/Work5_2/prompts_new/prompt_iiterative_check_step2_evaluation_Structure_Clarity_plain_order1.txt'
path_prompt_gen = r'/root/wpc/Work5_2/prompts_new/prompt_iterative_check_step3_related_work_based_on_structure_clarity_order1.txt'
with open(path_prompt_gen,'r',encoding='utf-8') as f:
    prompt_gen = f.read()

with open(path_prompt_eval,'r',encoding='utf-8') as f:
    prompt_eval = f.read()

system_message_eval = {"role": "system", "content": prompt_eval}
system_message_gen = {"role": "system", "content": prompt_gen}


path_api = r'/root/wpc/Work5/API_KEYS_4'
api_list = get_api_keys(path_api)

path_temp = r'/root/wpc/Work5/temp.txt'


path_input_text = r'/root/wpc/Work5_2/dataset/complete_related_work_dataset30.json'
path_input_meta = r'/root/wpc/Work5_2/dataset/meta_element30.json'
path_input_0sum = r'/root/wpc/Work5_2/result_new/gpt4_iterative_citation_complete30.txt'

path_w_temp = r'/root/wpc/Work5_2/result_new/gpt4_eval_temp.json'

path_w_json_temp = r'/root/wpc/Work5_2/result_new/gpt4_structure_clarity_temp.json'

path_w = r'/root/wpc/Work5_2/result_new/gpt4_meta_iterative_related_work_basedon_structure_clarity_order_total.json'
results = []
with open(path_input_text,'r',encoding='utf-8') as g_text, open(path_input_meta,'r',encoding='utf-8') as g_meta, \
    open(path_input_0sum,'r',encoding='utf-8') as g_0sum,  open(path_w,'a',encoding='utf-8') as h, open(path_w_json_temp,'a',encoding='utf-8') as h_temp:
    lines_text = g_text.readlines()
    lines_meta = g_meta.readlines()
    lines_0sum = g_0sum.readlines()
    # lines_eval = g_eval.readlines()

    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    api_index = 0

    start = 0
    end = 30
    for line_text, line_meta, line_0sum in zip(lines_text[start:end],lines_meta[start:end], lines_0sum[start:end]):

        line = json.loads(line_text)

        line_meta = json.loads(line_meta)

        del line_meta['id']

        total_result = {}
        now_draft = line_0sum.strip()
        total_result = {}
        for loop_step in range(3):
            openai.api_key = api_list[api_index]
            dic = {}
            message_gen = []
            message_eval = []

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

            dic['Related Work Draft'] = {}

            sent = nltk.sent_tokenize(now_draft)

            for index,s in enumerate(sent):
                dic["Related Work Draft"]["<SENTENCE_"+str(index+1)+">"] = s

            # dic['Feedback from the Review'] = line_eval

            # pdb.set_trace()
            with open(path_temp,'w',encoding='utf-8') as f:
                f.write(json.dumps(dic, ensure_ascii=False, indent=1))
                f.write('\n')

            with open(path_temp, 'r', encoding='utf-8') as f:

                inputs = f.read()

            user_prompt = 'Now let us begin, I will give you the input:\nInput:\n' + inputs

            user_prompt += '\nNow please generate the output in the required JSON format:\nOutput:\n'

            user_message = {"role": "user", "content":user_prompt}
            message_eval.append(system_message_eval) 
            message_eval.append(user_message) 

            output_eval =evaluator(message_eval)
            print("step "+str(loop_step)+ ' eval completed')
            # pdb.set_trace()

            output_eval = output_eval.replace('\n',' ').replace("```json",'').replace("```",'')

            with open(path_w_temp,'w',encoding='utf-8') as ff:
                ff.write(output_eval)
                ff.write('\n')
                ff.flush()

            with open(path_w_temp, 'r', encoding='utf-8') as ff:
                output_eval = ff.readlines()

            output_eval = json.loads(output_eval[0])
            total_result["step "+str(loop_step+1)+"_eval"] = output_eval

            h_temp.write(json.dumps(total_result))
            h_temp.write('\n')

            h_temp.flush()

            dic['Feedback from the Review'] = output_eval

            with open(path_temp,'w',encoding='utf-8') as ff:
                ff.write(json.dumps(dic, ensure_ascii=False, indent=1))
                ff.write('\n')

            with open(path_temp, 'r', encoding='utf-8') as ff:
                inputs = ff.read()

            user_prompt = 'Now let us begin, I will give you the input:\nInput:\n' + inputs

            user_prompt += '\nNow please generate the output in the required JSON format:\nOutput:\n'

            user_message = {"role": "user", "content":user_prompt}
            message_gen.append(system_message_gen) 
            message_gen.append(user_message) 

            output_gen =generator(message_gen)
            output_gen = output_gen.replace('\n',' ').replace("```json",'').replace("```",'')

            print("step "+str(loop_step)+ ' gen completed')

            with open(path_temp,'w',encoding='utf-8') as ff:
                ff.write(output_gen)
                ff.write('\n')

            with open(path_temp, 'r', encoding='utf-8') as ff:
                inputs = ff.readlines()

            output_gen_dict = json.loads(inputs[0])
            total_result["step "+str(loop_step+1)+"_gen"] = output_gen_dict

            h_temp.write(json.dumps(total_result))
            h_temp.write('\n')

            h_temp.flush()

            now_draft = ''
            for key in output_gen_dict["revised related work"]:
                now_draft += output_gen_dict["revised related work"][key]
                now_draft += ' '

            if api_index == len(api_list)-1:
                api_index = 0
            else:
                api_index += 1

        with open(path_w,'a',encoding='utf-8') as tt:
            tt.write(json.dumps(total_result))
            tt.write('\n')

            tt.flush()