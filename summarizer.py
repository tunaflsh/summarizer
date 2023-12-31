#!/usr/bin/env python
import argparse
import json
import os
import re
from types import SimpleNamespace

import prompt
from model import TOKEN_LIMIT, Model, trace


class Summarizer(Model):
    MARKDOWN_RULES = re.compile(r"\n\n *([-*_])( *\1){2,} *\n")

    def __init__(
        self,
        model="gpt-3.5-turbo",
        num_choices=1,
        checkpoint="summarizer.pkl",
        genre="detailed textbook",
        topic="[not specified]",
        language="English",
        context="",
        **kwargs,
    ):
        super().__init__(model, checkpoint, **kwargs)
        self.num_choices = num_choices
        self.genre = genre
        self.topic = topic
        self.language = language
        self.context = context
        self.state = {
            "__call__": SimpleNamespace(
                chunks=None, extracted=None, notes=None, final_notes=None
            ),
            "merge": SimpleNamespace(
                chunks=None, delimiter=None, merged=[], next_chunk="", i=0
            ),
            "extract_information": SimpleNamespace(chunks=None, compress=None, i=0),
            "compress": SimpleNamespace(notes=None, compressed=None),
            "write_final_text": SimpleNamespace(final_notes=None),
        }

    @trace
    def __call__(self, chunks=None):
        self.log("Start summarizing.", force=True)
        state = self.state["__call__"]

        if not (chunks or state.chunks):  # chunks are empty
            self.raise_error(ValueError("chunks are empty"))

        # Phase 0: Merge chunks of the original text
        if not state.chunks:  # if not from checkpoint
            self.log(
                f"Phase 0: Merging {len(chunks)} chunks of the original text.",
                force=True,
            )
            state.chunks = self.merge(chunks)
        if len(state.chunks) == 1:  # the original text is short
            self.log("Returning the short original text.", force=True)
            return state.chunks[0]

        # Phase 1: Extract information in form of notes
        self.log("Phase 1: Extracting information in form of notes.", force=True)
        if not state.extracted:  # if not from checkpoint
            state.extracted = self.extract_information(state.chunks)
        if not state.notes:  # if not from checkpoint
            state.notes = self.merge(state.extracted, delimiter="\n\n---\n\n")

        # Phase 2: Compress the notes
        if not state.final_notes:  # if not from checkpoint
            if len(state.notes) > 1:  # compress the notes
                self.log("Phase 2: Compressing the notes.", force=True)
                state.final_notes = self.compress(state.notes)
            else:  # the notes are short
                self.log("Skipping Phase 2 (Compression).", force=True)
                state.final_notes = state.notes[0]

        # Phase 3: Write the final text
        self.log("Phase 3: Write the final text.", force=True)
        final_text = self.write_final_text(state.final_notes)

        self.log("Finished summarizing.", force=True)
        # reset state
        state.chunks = None
        state.extracted = None
        state.notes = None
        return final_text

    @trace
    def merge(self, chunks=None, delimiter=""):
        state = self.state["merge"]
        # try to load from checkpoint
        state.chunks = state.chunks or chunks
        if state.delimiter is None:
            state.delimiter = delimiter
        self.log(f"Merging {len(state.chunks)} chunks.")

        if not state.chunks:  # chunks are empty
            self.raise_error(ValueError("chunks are empty"))

        for state.i in range(state.i, len(state.chunks)):
            chunk = state.chunks[state.i]
            if (tokens := len(self.tokenizer.encode(chunk))) > self.limit:
                # chunk too long
                self.raise_error(
                    ValueError(
                        f"chunk too long:"
                        f"    chunks[{state.i}] is {tokens} tokens long.\n"
                        f"    Limit is {self.limit} for {self.model}."
                    )
                )
            current_chunk = state.next_chunk
            state.next_chunk = (
                state.delimiter.join([current_chunk, chunk]) if current_chunk else chunk
            )
            if len(self.tokenizer.encode(state.next_chunk)) > self.limit:
                # current_chunk is full
                state.merged.append(current_chunk)
                state.next_chunk = chunk
        state.merged.append(state.next_chunk)

        chunks = state.merged
        self.log(f"{len(state.chunks)} -> {len(chunks)} chunks", force=True)

        # reset state
        state.chunks = None
        state.delimiter = None
        state.merged = []
        state.next_chunk = ""
        state.i = 0
        return chunks

    @trace
    def extract_information(self, chunks=None, compress=False):
        state = self.state["extract_information"]
        # try to load from checkpoint
        state.chunks = state.chunks or chunks
        state.compress = state.compress if state.compress is not None else compress

        messages = [
            {
                "role": "system",
                "content": (
                    prompt.EXTRACT.format(topic=self.topic, language=self.language)
                    + (
                        "\n\n" + prompt.CONTEXT.format(context=self.context)
                        if self.context
                        else ""
                    )
                    if not state.compress
                    else prompt.COMPRESS.format(
                        topic=self.topic, language=self.language
                    )
                ),
            },
            {"role": "user"},
        ]
        for state.i in range(state.i, len(state.chunks)):
            self.log(f"extract chunk {state.i+1}/{len(state.chunks)}:")
            messages[-1]["content"] = state.chunks[state.i]
            response = self.get_response(messages).choices[0].message.content
            response = self.MARKDOWN_RULES.sub("\n\n\n", response)
            state.chunks[state.i] = response

        chunks = state.chunks

        # reset state
        state.chunks = None
        state.compress = None
        state.i = 0
        return chunks

    @trace
    def compress(self, notes=None):
        state = self.state["compress"]
        state.notes = state.notes or notes  # try to load from checkpoint

        while len(state.notes) > 1:
            self.log(f"Compressing {len(state.notes)} chunks.")
            # try to load from checkpoint or compress notes
            if not state.compressed:
                state.compressed = self.extract_information(state.notes, compress=True)
            # try to load from checkpoint or merge notes
            state.notes = self.merge(state.compressed, delimiter="\n\n---\n\n")
            # reset state
            state.compressed = None

        notes = state.notes[0]

        # reset state
        state.notes = None
        state.compressed = None
        return notes

    @trace
    def write_final_text(self, final_notes=None):
        state = self.state["write_final_text"]
        state.final_notes = state.final_notes or final_notes

        messages = [
            {
                "role": "system",
                "content": prompt.WRITE.format(
                    genre=self.genre, topic=self.topic, language=self.language
                ),
            },
            {"role": "user", "content": state.final_notes + "\n\nFinal Text.md:"},
        ]
        response = self.get_response(messages, num_choices=self.num_choices)
        # longest choice & finish_reason=='stop'
        best_choice = max(
            response.choices,
            key=lambda c: len(c.message.content) if c.finish_reason == "stop" else 0,
        )

        # reset state
        state.final_notes = None
        return best_choice.message.content

    @classmethod
    def load(cls, checkpoint) -> "Summarizer":
        return super().load(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarizes long texts recursively with OpenAI API."
    )
    parser.add_argument("-i", "--input", type=str, help="Path to the input file")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file")
    parser.add_argument("-d", "--dir", type=str, help="Path to the output directory. Use with -i if -o is not specified.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=f'Model to use. Default: gpt-3.5-turbo. Options: {", ".join(list(TOKEN_LIMIT))}',
    )
    parser.add_argument(
        "-n",
        dest="num_choices",
        type=int,
        help="Number of times the final text will be generated. The longest complete text will be chosen. Default: 1",
    )
    parser.add_argument(
        "-g",
        "--genre",
        type=str,
        help="Genre or style of the text. Default: post. Examples: post, (detailed) textbook, essay, novel, scientific paper, script, etc.",
    )
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        help="Topic of the text. Default: [not specified]",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="Language of the text. Default: English",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        help="Something the model should keep in mind. Optional",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the log to stdout. Default: False",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="summarizer.pkl",
        help="Path to the checkpoint file. Default: summarizer.pkl",
    )
    parser.add_argument(
        "--load", action="store_true", help="Load the checkpoint file. Default: False"
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite the the final text from the checkpoint. Default: False",
    )
    args = parser.parse_args()

    if not (
        (args.input and args.dir)
        or (args.output and args.load)
        or (args.output and args.rewrite)
    ):
        parser.error(
            "Either [--input and --dir] or [--output and --load] or [--output and --rewrite] must be specified."
        )

    if args.output:
        output = args.output
    else:
        input, ext = os.path.splitext(os.path.basename(args.input))
        output = os.path.join(args.dir, input + ".md")

    # load checkpoint
    if args.load:
        if args.verbose:
            print(f"Loading checkpoint from {args.checkpoint}.")
        summarizer = Summarizer.load(args.checkpoint)
        summary = summarizer()

    # rewrite final text
    elif args.rewrite:
        if args.verbose:
            print(f"Loading checkpoint from {args.checkpoint}.")
        summarizer = Summarizer.load(args.checkpoint)
        summarizer.log("Rewriting final text.", force=True)
        # update number of choices n and genre
        summarizer.num_choices = args.num_choices or summarizer.num_choices
        summarizer.genre = args.genre or summarizer.genre
        summary = summarizer.write_final_text(summarizer.state["__call__"].final_notes)

    # generate summary
    else:
        summarizer = Summarizer(
            model=args.model or "gpt-3.5-turbo",
            num_choices=args.num_choices or 1,
            checkpoint=args.checkpoint or "summarizer.pkl",
            genre=args.genre or "detailed textbook",
            topic=args.topic or "[not specified]",
            language=args.language or "English",
            context=args.context or "",
            verbose=args.verbose,
        )
        with open(args.input, "r") as f:
            if args.verbose:
                print(f"Loading transcript from {args.input}.")
            transcript = json.load(f)
        chunks = [segment["text"] for segment in transcript["segments"]]
        summary = summarizer(chunks)

    # save summary
    with open(output, "w") as f:
        if args.verbose:
            print(f"Saving summary to {output}.")
        f.write(summary)

    print("Summary:", summary, sep="\n")
