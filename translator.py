#!/usr/bin/env python
import argparse

from model import Model, TOKEN_LIMIT, trace
import prompt


def translate(
    text,
    source_language,
    target_language,
    model="gpt-3.5-turbo",
    checkpoint="translator.pkl",
    **kwargs,
):
    model = Model(model, checkpoint, **kwargs)
    messages = [
        {
            "role": "system",
            "content": prompt.TRANSLATE.format(
                source_language=source_language, target_language=target_language
            ),
        },
        {"role": "user", "content": text},
    ]
    model.log("Translating...", force=True)
    response = model.get_response(messages)
    model.log("Finished translating.", force=True)
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate a text file from one language to another."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The path to the text file to translate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The path to save the translated text file to.",
    )
    parser.add_argument(
        "-s",
        "--source-language",
        type=str,
        required=True,
        help="The language of the text file to translate.",
    )
    parser.add_argument(
        "-t",
        "--target-language",
        type=str,
        required=True,
        help="The language to translate the text file to.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help=f'The model to use for translation. Default: gpt-3.5-turbo. Options: {", ".join(list(TOKEN_LIMIT))}',
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="translator.pkl",
        help="The checkpoint to use for the model.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print the logs to stdout.",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    translated_text = translate(
        text,
        args.source_language,
        args.target_language,
        args.model,
        args.checkpoint,
        verbose=args.verbose,
    )

    with open(args.output, "w") as f:
        f.write(translated_text)
