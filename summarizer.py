#!/usr/bin/env python
import argparse
import json
import re
from types import SimpleNamespace

import aiapi
import prompt


class Summarizer(aiapi.Model):
    MARKDOWN_RULES = re.compile(r'\n\n *([-*_])( *\1){2,} *\n')

    def __init__(self, model='gpt-3.5-turbo', n=1, checkpoint='summarizer.pkl',
                 genre='detailed textbook', topic='[not specified]', language='English',
                 **kwargs):
        super().__init__(model, checkpoint, **kwargs)
        self.n = n
        self.genre = genre
        self.topic = topic
        self.language = language
        self.state = {
            '__call__': SimpleNamespace(
                chunks=None,
                extracted=None,
                notes=None,
                final_notes=None),
            'merge': SimpleNamespace(
                chunks=None,
                delimiter=None,
                merged=[],
                next_chunk='',
                i=0),
            'extract_information': SimpleNamespace(
                chunks=None,
                compress=None,
                i=0),
            'compress': SimpleNamespace(
                notes=None,
                compressed=None),
            'write_final_text': SimpleNamespace(
                final_notes=None)
        }
        self.stack_trace = []

    @staticmethod
    def _trace(func):
        def wrapper(self, *args, **kwargs):
            self.stack_trace.append(func.__name__)
            result = func(self, *args, **kwargs)
            self.stack_trace.pop()
            self.save()
            return result
        return wrapper

    @_trace
    def __call__(self, chunks=None):
        self.log('Start summarizing.', force=True)
        state = self.state['__call__']

        if not (chunks or state.chunks):  # chunks are empty
            self.raise_error(ValueError('chunks are empty'))
        
        # Phase 0: Merge chunks of the original text
        if not state.chunks:  # if not from checkpoint
            self.log(f'Phase 0: Merging {len(chunks)} chunks of the original text.',
                     force=True)
            state.chunks = self.merge(chunks)
        if len(state.chunks) == 1:  # the original text is short
            self.log('Returning the short original text.', force=True)
            return state.chunks[0]

        # Phase 1: Extract information in form of notes
        self.log(f'Phase 1: Extracting information in form of notes.', force=True)
        if not state.extracted:  # if not from checkpoint
            state.extracted = self.extract_information(state.chunks)
        if not state.notes:  # if not from checkpoint
            state.notes = self.merge(state.extracted, delimiter='\n\n---\n\n')

        # Phase 2: Compress the notes
        if not state.final_notes:  # if not from checkpoint
            if len(state.notes) > 1: # compress the notes
                self.log(f'Phase 2: Compressing the notes.', force=True)
                state.final_notes = self.compress(state.notes)
            else:  # the notes are short
                self.log('Skipping Phase 2 (Compression).', force=True)
                state.final_notes = state.notes[0]

        # Phase 3: Write the final text
        self.log('Phase 3: Write the final text.', force=True)
        final_text = self.write_final_text(state.final_notes)
        
        self.log('Finished summarizing.', force=True)
        # reset state
        state.chunks = None
        state.extracted = None
        state.notes = None
        return final_text

    @_trace
    def merge(self, chunks=None, delimiter=''):
        state = self.state['merge']
        # try to load from checkpoint
        state.chunks = state.chunks or chunks
        if state.delimiter is None:
            state.delimiter = delimiter
        self.log(f'Merging {len(state.chunks)} chunks.')

        if not state.chunks:  # chunks are empty
            self.raise_error(ValueError('chunks are empty'))

        for state.i in range(state.i, len(state.chunks)):
            chunk = state.chunks[state.i]
            if (tokens := len(self.tokenizer.encode(chunk))) > self.limit:
                # chunk too long
                self.raise_error(ValueError(
                    f'chunk too long:'
                    f'    chunks[{state.i}] is {tokens} tokens long.\n'
                    f'    Limit is {self.limit} for {self.model}.'))
            current_chunk = state.next_chunk
            state.next_chunk = state.delimiter.join([current_chunk, chunk]) \
                if current_chunk else chunk
            if len(self.tokenizer.encode(state.next_chunk)) > self.limit:
                # current_chunk is full
                state.merged.append(current_chunk)
                state.next_chunk = chunk
        state.merged.append(state.next_chunk)

        chunks = state.merged
        self.log(f'{len(state.chunks)} -> {len(chunks)} chunks', force=True)

        # reset state
        state.chunks = None
        state.delimiter = None
        state.merged = []
        state.next_chunk = ''
        state.i = 0
        return chunks
    
    @_trace
    def extract_information(self, chunks=None, compress=False):
        state = self.state['extract_information']
        # try to load from checkpoint
        state.chunks = state.chunks or chunks
        state.compress = state.compress if state.compress is not None else compress

        messages = [
            {'role': 'system', 'content': (
                prompt.COMPRESS if state.compress else prompt.EXTRACT
                ).format(topic=self.topic, language=self.language)},
            {'role': 'user'}
        ]
        for state.i in range(state.i, len(state.chunks)):
            self.log(f'extract chunk {state.i+1}/{len(state.chunks)}:')
            messages[-1]['content'] = state.chunks[state.i]
            response = self.get_response(messages).choices[0].message.content
            response = self.MARKDOWN_RULES.sub('\n\n\n', response)
            state.chunks[state.i] = response
        
        chunks = state.chunks

        # reset state
        state.chunks = None
        state.compress = None
        state.i = 0
        return chunks

    @_trace
    def compress(self, notes=None):
        state = self.state['compress']
        state.notes = state.notes or notes  # try to load from checkpoint

        while len(state.notes) > 1:
            self.log(f'Compressing {len(state.notes)} chunks.')
            # try to load from checkpoint or compress notes
            if not state.compressed:
                state.compressed = self.extract_information(state.notes,
                                                            compress=True)
            # try to load from checkpoint or merge notes
            state.notes = self.merge(state.compressed, delimiter='\n\n---\n\n')
            # reset state
            state.compressed = None
        
        notes = state.notes[0]

        # reset state
        state.notes = None
        state.compressed = None
        return notes

    @_trace
    def write_final_text(self, final_notes=None):
        state = self.state['write_final_text']
        state.final_notes = state.final_notes or final_notes

        messages = [
            {'role': 'system', 'content':
                prompt.WRITE.format(
                    genre=self.genre,
                    topic=self.topic,
                    language=self.language)},
            {'role': 'user', 'content':
                state.final_notes
                + '\n\nFinal Text.md:'},
        ]
        response = self.get_response(messages, n=self.n)
        # longest choice & finish_reason=='stop'
        best_choice = max(response.choices,
                          key=lambda c: len(c.message.content) \
                            if c.finish_reason == 'stop' else 0)
        
        # reset state
        state.final_notes = None
        return best_choice.message.content


if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the input file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output file')
    parser.add_argument('-m', '--model', type=str,
                        help=f'Model to use. Default: gpt-3.5-turbo. Options: {", ".join(list(aiapi.TOKEN_LIMIT))}')
    parser.add_argument('-n', dest='choices', type=int,
                        help='Number of completion choices to generate for each input message. Default: 1')
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='Path to the checkpoint file. Default: summarizer.pkl')
    parser.add_argument('-g', '--genre', type=str,
                        help='Genre of the text. Default: detailed textbook. Examples: textbook, essay, novel, scientific paper, script, etc.')
    parser.add_argument('-t', '--topic', type=str,
                        help='Topic of the conversation. Default: [not specified]')
    parser.add_argument('-l', '--language', type=str,
                        help='Language of the conversation. Default: English')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print the log to stdout. Default: False')
    parser.add_argument('--load', action='store_true',
                        help='Load the checkpoint file. Default: False')
    parser.add_argument('--rewrite', action='store_true',
                        help='Rewrite the the final text from the checkpoint. Default: False')
    args = parser.parse_args()

    # load checkpoint
    if args.load:
        if args.verbose:
            print(f'Loading checkpoint from {args.checkpoint}.')
        summarizer = Summarizer.load(args.checkpoint)
        # override arguments
        summarizer.model = args.model or summarizer.model
        summarizer.choices = args.choices or summarizer.choices
        summarizer.genre = args.genre or summarizer.genre
        summarizer.topic = args.topic or summarizer.topic
        summarizer.language = args.language or summarizer.language
        summary = summarizer()

    # rewrite final text
    elif args.rewrite:
        if args.verbose:
            print(f'Loading checkpoint from {args.checkpoint}.')
        summarizer = Summarizer.load(args.checkpoint)
        # override arguments
        summarizer.model = args.model or summarizer.model
        summarizer.choices = args.choices or summarizer.choices
        summarizer.genre = args.genre or summarizer.genre
        summarizer.topic = args.topic or summarizer.topic
        summarizer.language = args.language or summarizer.language
        summarizer.log(f'Rewriting final text.', force=True)
        summary = summarizer.write_final_text(
            summarizer.state['__call__'].final_notes)
    
    # generate summary
    else:
        summarizer = Summarizer(model=args.model or 'gpt-3.5-turbo',
                                n=args.choices or 1,
                                checkpoint=args.checkpoint or 'summarizer.pkl',
                                genre=args.genre or 'detailed textbook',
                                topic=args.topic or '[not specified]',
                                language=args.language or 'English',
                                verbose=args.verbose)
        with open(args.input, 'r') as f:
            if args.verbose:
                print(f'Loading transcript from {args.input}.')
            transcript = json.load(f)
        chunks = [segment['text'] for segment in transcript['segments']]
        summary = summarizer(chunks)
    
    # save summary
    with open(args.output, 'w') as f:
        if args.verbose:
            print(f'Saving summary to {args.output}.')
        f.write(summary)
    
    print('Summary:', summary, sep='\n')
