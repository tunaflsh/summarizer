# Summarizer

Summarizes long texts recursively with OpenAI API. This summarizer can be configured to output in a specified style (see `--genre`) and language (see `--language`), and can be hinted with the topic (see `--topic`). Any additional information can be provided in `--context`.

To prevent information losses from intermediate summaries, the model is instructed to ***take notes*** of the text chunks. This way details and facts are preserved while the notes are more compact than the original text. The final summary then combines all the notes into a coherent text.

*Currently the summarizer only accept inputs as `json` verbose transcripts from Whisper API (you may be interested [whisper-youtube-transcriber](https://github.com/tunaflsh/whisper-youtube-transcriber)). In the future I plan to make it summarize plain texts as well.*

## Usage

```
python summarizer.py [-h] [-i INPUT] [-o OUTPUT] [-d DIR] [-m MODEL] [-n NUM_CHOICES] [-g GENRE] [-t TOPIC]
                     [-l LANGUAGE] [-c CONTEXT] [-v] [--checkpoint CHECKPOINT] [--load] [--rewrite]

Summarizes long texts recursively with OpenAI API.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input file
  -o OUTPUT, --output OUTPUT
                        Path to the output file
  -d DIR, --dir DIR     Path to the output directory. Use with -i if -o is not specified.
  -m MODEL, --model MODEL
                        Model to use. Default: gpt-3.5-turbo. Options: gpt-4, gpt-3.5-turbo-16k, gpt-3.5-turbo,
                        gpt-4-0613, gpt-4-0314, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0613, gpt-3.5-turbo-0301
  -n NUM_CHOICES        Number of times the final text will be generated. The longest complete text will be chosen.
                        Default: 1
  -g GENRE, --genre GENRE
                        Genre or style of the text. Default: post. Examples: post, (detailed) textbook, essay, novel,
                        scientific paper, script, etc.
  -t TOPIC, --topic TOPIC
                        Topic of the text. Default: [not specified]
  -l LANGUAGE, --language LANGUAGE
                        Language of the text. Default: English
  -c CONTEXT, --context CONTEXT
                        Something the model should keep in mind. Optional
  -v, --verbose         Print the log to stdout. Default: False
  --checkpoint CHECKPOINT
                        Path to the checkpoint file. Default: summarizer.pkl
  --load                Load the checkpoint file. Default: False
  --rewrite             Rewrite the the final text from the checkpoint. Default: False
```

### Output Directory

The summarizer now supports an optional argument `-d` to specify the output directory for the summary file. If this argument is provided, the summary file will be saved in the specified directory with the same name as the `input` file and a `.md` extension. If this argument is not provided, the `output` argument is required to specify the output file path.

To use this argument, simply add `-d` followed by the path to the output directory when running the script. For example:

```shell
python summarizer.py -i input.json -d output/
```

This will save the final summary in `output/input.md`.

### Rewriting Final Summary

Since the intermediate summaries are notes and only the last summary is a coherent text, the genre or style is only applied to that last stage. The final summary can be rewritten with the same notes from intermediate summaries. The options `-n` and `--genre` can be modified in this mode:
```
python summarizer.py -o OUTPUT [-n NUM_CHOICES] [-g GENRE] [--checkpoint CHECKPOINT] --rewrite
```

### Continuing from Checkpoint

In case of unexpected crashes, the summarization process can be continued from a checkpoint:
```
python summarizer.py -o OUTPUT [--checkpoint CHECKPOINT] --load
```

## Logs

Logs are stored in the `logs/` folder. They are useful for debugging and looking at intermediate summaries.
