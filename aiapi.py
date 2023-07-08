from datetime import datetime
import json
import pickle
import openai
import tiktoken


TOKEN_LIMIT = {
    'gpt-4': 5000,
    'gpt-3.5-turbo-16k': 10000,
    'gpt-3.5-turbo': 2000,
    'gpt-4-0613': 5000,
    'gpt-4-0314': 5000,
    'gpt-3.5-turbo-16k-0613': 10000,
    'gpt-3.5-turbo-0613': 2000,
    'gpt-3.5-turbo-0301': 2000
}


class Logger:
    def __init__(self, log_file='./logs/{day}.log', verbose=False):
        self.log_file = log_file.format(day=datetime.now().strftime("%Y-%m-%d"))
        self.verbose = verbose

    def log(self, *message, force=False):
        with open(self.log_file, 'a') as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, m in enumerate(message):
                log_message = f'{current_time}: {m}\n' \
                                if i == 0 else \
                                f'{" " * len(current_time)}  {m}\n'
                f.write(log_message)
                if force or self.verbose:
                    print(log_message, end='')


class Model:
    def __init__(self, model, checkpoint_path, **kwargs):
        self.model = model
        self.limit = TOKEN_LIMIT[model]
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.logger = Logger(**kwargs)
        self.checkpoint_path = checkpoint_path
    
    def get_response(self, messages, **kwargs):
        self.logger.log('Requesting:',
                        *[f'{role}: {content}'
                          for role, content in messages.items()])
        self.logger.log('Waiting for response...')
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages)
        except openai.OpenAIError as e:
            self.raise_error(e, **kwargs)
        self.logger.log('Response:',
                        f'model: {self.model}',
                        f'{response.choices[0].message.assistant}'
                        f': {response.choices[0].message.content}',
                        f'finish_reason: {response.choices[0].finish_reason}',
                        f'usage: {json.dumps(response.usage)}')
        return response
    
    def raise_error(self, e, **kwargs):
        self.logger.log(e)
        self.save_checkpoint(**kwargs)
        raise e

    def save_checkpoint(self, checkpoint_path=None):
        with open(checkpoint_path or self.checkpoint_path, 'w') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return pickle.load(f)
