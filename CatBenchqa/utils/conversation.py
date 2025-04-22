import os
import tiktoken
import anthropic
import time
import boto3
import json
import ollama
import subprocess
from botocore.exceptions import ClientError
from openai import OpenAI
import pickle as pkl 
from ratelimit import limits, sleep_and_retry
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


# llm_models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o, claude-3-haiku-20240307 (small), claude-3-sonnet-20240229 (medium), claude-3-opus-20240229 (large), meta.llama3-8b-instruct-v1:0, meta.llama3-1-8b-instruct-v1:0

class Conversation:
    def __init__(self, llm_model, temp) -> None:
        self.llm_prompt = []
        self.log_history = []
        self.llm_model =  llm_model
        self.temp = temp
        self.tokens_per_min = 0
        self.max_tokens = 256
        self.input_token_cost = 0.5 / 1e6 # only for gpt-3.5-turbo
        self.output_token_cost = 1.5 / 1e6
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.cost_limit = 10
        self._setup_client(llm_model)
        
    def _setup_client(self, llm_model):
        if llm_model == "gpt-3.5-turbo" or llm_model == "gpt-4o-mini" or llm_model == "gpt-4o":
            api_key = os.environ["OPENAI_API_KEY"]
            # key_file = open(os.getcwd()+'/key.txt', 'r')
            # api_key = key_file.readline().rstrip()

            if api_key is None:
                raise Exception("Please insert your OpenAI API key in conversation.py")

            self.client = OpenAI(
            api_key=api_key,
            )
            
        elif 'claude' in llm_model:
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            self.client = self.anthropic_client
            self.max_tokens_per_min = 50000 #TODO: get this from the API
            self.max_tokens_per_day = 1000000
            
        # elif 'llama3-1' in llm_model:
        #     self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
        # elif 'llama3' in llm_model:
        #     self.client = boto3.client("bedrock-runtime", region_name="us-east-1")
        
        elif "llama" in llm_model:
            command_run = f"ollama start {llm_model}"
            subprocess.run(command_run, shell=True)

        
    def count_tokens(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
       
    def construct_message(self, prompt, role):
        assert role in ["user", "assistant"]
        new_message = {"role": role, "content": prompt}
        self.llm_prompt = []
        message = self.llm_prompt + [new_message]
        input_tokens = self.count_tokens(message[0]['content'],  'cl100k_base')
        self.total_cost += self.input_token_cost * input_tokens
        return message

    def get_response(self, prompt, stop=None, temperature=0, role="user"): 
        # chat model       
        
        message = self.construct_message(prompt, role) 
        
        if self.llm_model == "gpt-3.5-turbo" or self.llm_model == "gpt-4o-mini" or self.llm_model == "gpt-4o": 
        
            if self.total_cost > self.cost_limit:
                return {"response_message": "[WARNING] COST LIMIT REACHED!"}
            else:
                response = self.client.chat.completions.create(
                model=self.llm_model,
                messages = message,
                temperature=self.temp,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["END"]
                )

            answer = response.choices[0].message.content
            output_tokens = self.count_tokens(answer, 'cl100k_base')
            self.total_cost += self.output_token_cost * output_tokens
            
        elif 'claude' in self.llm_model:
        
            anthropic_client = self.anthropic_client
            local_config = {"max_tokens": 100, "temperature": self.temp}
            
            
            @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6))
            def get_claude_response():
                tokens_per_min = 0
                tokens_per_day = 0
                response = anthropic_client.messages.create(
                    model=self.llm_model,
                    max_tokens=local_config['max_tokens'],
                    temperature=local_config['temperature'],
                    messages=message
                )
                claude_input_tokens = response.usage.input_tokens
                self.total_input_tokens += claude_input_tokens
                claude_output_tokens = response.usage.output_tokens
                self.total_output_tokens += claude_output_tokens
                answer = response.content[0].text
                tokens_per_min += claude_input_tokens + claude_output_tokens
                tokens_per_day += claude_input_tokens + claude_output_tokens
                if tokens_per_min > self.max_tokens_per_min:
                    print("[INFO] Sleeping for 60 seconds to avoid rate limit.")
                    time.sleep(60)
                    tokens_per_min = 0
                if tokens_per_day > self.max_tokens_per_day:
                    print("[INFO] Sleeping for 24 hours to avoid rate limit.")
                    time.sleep(86400)
                    tokens_per_day = 0
                return answer, claude_input_tokens, claude_output_tokens
            
            answer, input_tokens, output_tokens = get_claude_response()
            
        elif 'llama' in self.llm_model:
            messages = [{"role": "user", "content": prompt}]
            response = ollama.chat(
                model=self.llm_model,
                messages=messages
            )
            content = response['message']['content']
            
            answer = content
            # formatted_prompt = f"""
            # <|begin_of_text|>
            # <|start_header_id|>user<|end_header_id|>
            # {prompt}
            # <|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>
            # """

            # # Format the request payload using the model's native structure.
            # native_request = {
            #     "prompt": formatted_prompt,
            #     "max_gen_len": 512,
            #     "temperature": self.temp,
            # }
            
            # request = json.dumps(native_request)
            # try:
            #     # Invoke the model with the request.
            #     response = self.client.invoke_model(modelId=self.llm_model, body=request)

            # except (ClientError, Exception) as e:
            #     print(f"ERROR: Can't invoke '{self.llm_model}'. Reason: {e}")
            #     exit(1)

            # # Decode the response body.
            # model_response = json.loads(response["body"].read())

            # # Extract and print the response text.
            # answer = model_response["generation"]
            # if answer[0] == ' ':
            #     answer = answer[1:]
        
        self.log_history.append(answer)
        self.llm_prompt.append(prompt + answer + "\n")
        return answer
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            object = {"prompt": self.llm_prompt, "log_history": self.log_history, "model": self.llm_model}
            pkl.dump(object, f)


if __name__ == "__main__":
    c = Conversation("gpt-3.5-turbo")
    PROMPT_1 = "Hello, how are you?"
    PROMPT_2 = "I've been better."

    print(f"User : {PROMPT_1}")
    x1 = c.get_response(PROMPT_1)
    print("LLM : ", x1['response_message'])
    
    print(f"User : {PROMPT_2}")
    x2 = c.get_response(PROMPT_2)
    print("LLM : ", x2['response_message'])

    print("done")