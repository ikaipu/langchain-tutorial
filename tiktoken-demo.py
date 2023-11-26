# !pip install tiktoken
import tiktoken

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
text = "This is a test for tiktoken."
tokens = encoding.encode(text)
print(len(text))  # 28
print(tokens)  # [2028, 374, 264, 1296, 369, 87272, 5963, 13]
print(len(tokens))  # 8