import prompt
import summarizer


if __name__ == '__main__':
    # Create a summarizer model.
    model = summarizer.Summarizer('gpt-3.5-turbo')

    # Create a prompt.
    prompt = prompt.Prompt()

    # Create a summarizer evaluator.
    evaluator = summarizer.Evaluator(model, prompt)

    # Evaluate the summarizer model.
    evaluator.evaluate()
