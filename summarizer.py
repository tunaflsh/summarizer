import argparse
from types import SimpleNamespace
import aiapi
import prompt


class Summarizer(aiapi.Model):
    def __init__(self,
                 model='gpt-3.5-turbo',
                 checkpoint_path='summarizer.pkl',
                 topic='unknown',
                 language='English',
                 **kwargs):
        super().__init__(model, checkpoint_path, **kwargs)
        self.topic = topic
        self.language = language
        self.state = {
            '__call__': SimpleNamespace(
                chunks=None,
                notes=None),
            'merge': SimpleNamespace(
                chunks=None,
                new_chunks=[],
                next_chunk='',
                i=0),
            'extract_information': SimpleNamespace(
                chunks=None,
                result=[],
                i=0),
            'reconstruct': SimpleNamespace(notes=None),
            'report': SimpleNamespace(notes=None)
        }
        self.stack_trace = []

    def trace(self, func):
        def wrapper(*args, **kwargs):
            self.stack_trace.append(func.__name__)
            result = func(*args, **kwargs)
            self.stack_trace.pop()
            return result
        return wrapper

    @trace
    def __call__(self, chunks=None):
        state = self.state['__call__']
        state.chunks = state.chunks or chunks
        self.logger.log(f'Summarize {len(state.chunks)} chunks.')
        state.notes = state.notes or self.extract_information(state.chunks, self.topic, self.language)
        state.result = self.reconstruct(state.notes, self.topic, self.language)
        self.logger.log(f'Finish summarizing.')
        result = state.result
        state.chunks = None
        state.notes = None
        state.result = None
        return result

    @trace
    def merge(self, chunks=None):
        state = self.state['merge']
        state.chunks = state.chunks or chunks
        self.logger.log(f'Merge {len(state.chunks)} chunks.')
        for state.i in range(state.i, len(state.chunks)):
            chunk = state.chunks[state.i]
            if tokens := len(self.tokenizer.encode(chunk)) > self.limit:
                self.raise_error(ValueError(
                    f'chunk too long:'
                    f'    chunks[{state.i}] is {tokens} tokens long.\n'
                    f'    Limit is {self.limit} for {self.model}.'))
            current_chunk = state.next_chunk
            state.next_chunk += chunk
            if len(self.tokenizer.encode(state.next_chunk)) > self.limit:
                state.new_chunks.append(current_chunk)
                state.next_chunk = chunk
        state.new_chunks.append(state.next_chunk)
        self.logger.log(f'Finish with {len(state.new_chunks)} chunks.')

        chunks = state.new_chunks
        state.chunks = None
        state.new_chunks = []
        state.next_chunk = ''
        state.i = 0
        return chunks

    @trace
    def extract_information(self, chunks=None):
        state = self.state['extract_information']
        state.chunks = state.chunks or chunks
        if len(state.chunks) == 1:
            self.logger.log('Phase 1: Return the only chunk.')
            return state.chunks[0]
        state.chunks = self.merge(state.chunks)
        self.logger.log(f'Phase 1: Extract information from {len(state.chunks)} chunks.')
        for state.i in range(state.i, len(state.chunks)):
            self.logger.log(f'Chunk {state.i+1}/{len(state.chunks)}:')
            messages = [{'role': 'system',
                         'content': prompt.PHASE1.format(topic=self.topic,
                                                         language=self.language)},
                        {'role': 'user', 'content': state.chunks[state.i]}]
            response = self.get_response(messages)
            state.result.append(response.choices[0].message.content)
        self.logger.log('Phase 1: Finish.')

        result = state.result
        state.chunks = None
        state.result = []
        state.i = 0
        return self.extract_information(result)
    
    @trace
    def reconstruct(self, notes=None):
        state = self.state['reconstruct']
        state.notes = state.notes or notes
        self.logger.log('Phase 2a: Reconstructing text from notes.')
        messages = [{'role': 'system',
                     'content': prompt.PHASE2A.format(topic=self.topic,
                                                      language=self.language)},
                    {'role': 'user', 'content': notes}]
        response = self.get_response(messages)
        self.logger.log('Phase 2a: Finish.')
        state.notes = None
        return response.choices[0].message.content

    @trace
    def report(self, notes=None):
        state = self.state['report']
        state.notes = state.notes or notes
        self.logger.log('Phase 2b: Creating a report from notes.')
        messages = [{'role': 'system',
                     'content': prompt.PHASE2B.format(topic=self.topic,
                                                      language=self.language)},
                    {'role': 'user', 'content': notes}]
        response = self.get_response(messages)
        self.logger.log('Phase 2b: Finish.')
        state.notes = None
        return response.choices[0].message.content


if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the input file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output file')
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo',
                        help=f'Model to use. Options: {list(aiapi.TOKEN_LIMIT)}. Default: gpt-3.5-turbo')
    parser.add_argument('-c', '--checkpoint', type=str, default='summarizer.pkl',
                        help='Path to the checkpoint file. Default: summarizer.pkl')
    parser.add_argument('-t', '--topic', type=str, default='unknown',
                        help='Topic of the conversation. Default: unknown')
    parser.add_argument('-l', '--language', type=str, default='English',
                        help='Language of the conversation. Default: English')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print the log to stdout. Default: False')

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        if args.verbose:
            print(f'Loading transcript from {args.input}.')
        transcript = f.read()
    chunks = [segment['text'] for segment in transcript['segments']]

    summarizer = Summarizer(args.model, args.checkpoint, args.topic, args.language)
    summary = summarizer(chunks)
    
    with open(args.output, 'w') as f:
        if args.verbose:
            print(f'Saving summary to {args.output}.')
        f.write(summary)
    
    print('Summary:', summary, sep='\n')
