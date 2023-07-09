EXTRACT = '''
Take notes from the text provided by the user on the topic "{topic}" in {language}. It should be structured using markdown and be densely packed with information so that the original text can be reconstructed from it.
'''.strip()

COMPRESS = '''
Compress the notes provided by the user on the topic "{topic}" in {language} into a shorter markdown formatted text. It should be lossless and contain all the information from the original notes.
'''.strip()

WRITE = '''
Write a markdown formatted {genre} in {language} on the topic "{topic}", that covers the entire content of the multiple notes seperated by horizontal rules provided by the user.
'''.strip()
