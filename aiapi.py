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
            for i, m in enumerate([m for ms in message for m in ms.split('\n')]):
                log_message = f'{current_time}: {m}\n' \
                                if i == 0 else \
                                f'{" " * len(current_time)}  {m}\n'
                f.write(log_message)
                if force or self.verbose:
                    print(log_message, end='')


class Model:
    def __init__(self, model, checkpoint, **kwargs):
        self.model = model
        self.limit = TOKEN_LIMIT[model]
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.logger = Logger(**kwargs)
        self.checkpoint = checkpoint
    
    def get_response(self, messages, n=1, **kwargs):
        self.logger.log('Requesting:',
                        *[f'{message["role"]}: {message["content"]}'
                          for message in messages])
        self.logger.log('Awaiting response...')
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                n=n,
                messages=messages)
        except openai.OpenAIError as e:
            self.raise_error(e, **kwargs)
        self.logger.log('Response:',
                        f'model: {self.model}')
        for i, choice in enumerate(response.choices):
            self.logger.log(f'choice[{i}]: {choice.message.role}:',
                            f'{choice.message.content}')
            self.logger.log(f'choice[{i}]: finish_reason: '
                            f'{choice.finish_reason}')
        self.logger.log(f'usage: {json.dumps(response.usage)}')
        return response
    
    def raise_error(self, e: ValueError, **kwargs):
        self.logger.log(str(e))
        self.save(**kwargs)
        raise e

    def save(self, checkpoint=None):
        del self.tokenizer
        with open(checkpoint or self.checkpoint, 'wb') as f:
            pickle.dump(self, f)
        self.tokenizer = tiktoken.encoding_for_model(self.model)
    
    @classmethod
    def load(cls, checkpoint):
        with open(checkpoint, 'rb') as f:
            model = pickle.load(f)
        model.tokenizer = tiktoken.encoding_for_model(model.model)
        return model
