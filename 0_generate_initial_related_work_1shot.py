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

@backoff.on_exception(backoff.expo, RateLimitError)
def generator(d):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        # model="gpt-4-turbo-preview",
        # model="gpt-4-0125-preview",
        messages = [{"role": "user", "content": d}],
        temperature=0,
        max_tokens= 4000,
    )
    ini_sent = response.choices[0]['message']['content']
    # response = completion_with_backoff(model="gpt-3.5-turbo-instruct", prompt=d, max_tokens=1500, temperature=0)
    # ini_sent = response["choices"][0]["text"]
    # import pdb; pdb.set_trace()
    # ini_sent = response.choices[0]['message']['content']

    return ini_sent    

# path_prompt = r'/root/wpc/Work5/prompts/prompt2_related_work_generation_0shot.txt')
path_prompt = r'/root/wpc/Work5/prompts/prompt_related_work_generation_fewshot.txt'
with open(path_prompt,'r',encoding='utf-8') as f:
    prompt = f.read()

path_temp = r'/root/wpc/Work5/temp.txt'

path_api = r'/root/wpc/Work5/API_KEYS_3.5'
api_list = get_api_keys(path_api)

path_input = r'/root/wpc/Work5/dataset/complete_related_work_dataset30.json'
path_w = r'/root/wpc/Work5/result/gpt35_complete_abstract_0shot_0exmp.json'
results = []

api_index = 0
with open(path_input,'r',encoding='utf-8') as g, open(path_w,'a',encoding='utf-8') as h:
    lines = g.readlines()
    openai.api_key = api_list[api_index]

    for line in lines:
        line = json.loads(line)

        dic = {}

        dic['Target Paper'] = {}
        dic['Target Paper']['Title'] = line['target_paper']['title']
        dic['Target Paper']['Abstract'] = line['target_paper']['abstract']

        references = line['reference']

        dic['Reference Papers'] = {}

        for ref in references:
            refnew = ref.strip()
            dic['Reference Papers'][refnew] = {}
            dic['Reference Papers'][refnew]['Title'] = references[ref]['title']
            dic['Reference Papers'][refnew]['Abstract'] = references[ref]['abstract']


        with open(path_temp,'w',encoding='utf-8') as f:
            f.write(json.dumps(dic, ensure_ascii=False, indent=1))
            f.write('\n')

        with open(path_temp, 'r', encoding='utf-8') as f:

            inputs = f.read()

            now_prompt = prompt + '\n' + inputs

            now_prompt += '\nNow please generate the related work section directly. \n'

            output =generator(now_prompt)

            output = output.replace('\n',' ')


            h.write(output)
            h.write('\n')
            
            h.flush()

            if api_index == len(api_list)-1:
                api_index = 0
            else:
                api_index += 1