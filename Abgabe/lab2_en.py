#!/usr/bin/env python
# coding: utf-8

# # University Project: Large Language Models from Scratch
# **Group:** Phillip Graf, Konstantin Schmidt, Fabian Holländer
# 
# ## Lab 2: Tokenization and positional embeddings
# In this lab, we import our dataset and perform the necessary preprocessing steps. We then focus on generating tokens from the recipe data and, as a final step, implement positional embeddings to prepare the sequences for the model.
# 
# For learning and testing purposes we import a reduced dataset of 100.000 reciepes.
# https://syncandshare.lrz.de/dl/fiHE8nDPcb4nww3VCn4QmN/reduced_dataset_100k.csv
# 
# *Optional* the full dataset containing 2.2 million recipes can be used as input for larger-scale training.
# https://syncandshare.lrz.de/dl/fiHE8nDPcb4nww3VCn4QmN/full_dataset.csv

# ## Import necessary libaries

# In[71]:


import pandas as pd
import re
import importlib
import tiktoken
import torch


# ## Import the dataset from the cloud

# In[72]:


# Reduced dataset with 100k rows for testing
cloud_url = "https://syncandshare.lrz.de/dl/fiHE8nDPcb4nww3VCn4QmN/reduced_dataset_100k.csv"
# Uncomment the following line to use the full dataset
# cloud_url = "https://syncandshare.lrz.de/dl/fiHE8nDPcb4nww3VCn4QmN/full_dataset.csv"

try:
    print("Loading dataset from cloud...")
    df = pd.read_csv(cloud_url)
    print("Dataset loaded successfully!\n")
    print("Info")
    print(df.info())
    print("")
    print("Head of the dataset:")
    print(df.head())
    
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")


# ## Formatting of the dataset to one input string
# This allows us to deduplicate the vocabulary, generate consistent tokens, and significantly streamline sequence iteration later in the process.
# 
# Since only the three columns **title**, **ingredients**, and **directions** are necessary for text understanding, we use only these to generate our concatenated input string.

# In[73]:


def format_csv(row):
    title = str(row['title'])
    ingredients = str(row['ingredients']).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    directions = str(row['directions']).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    return f"Recepie: {title}\nIngredients: {ingredients}\nDirections: {directions}"

raw_text = "".join(df.apply(format_csv, axis=1))
print(f"The entire dataset has been successfully converted into a single string.")
print(f"Total character count: {len(raw_text)}")
print("\nPreview of the first 100 characters:")
print(raw_text[:100])

# TODO -> umstellen auf full raw text?
# For a faster computation we only select the first 100 characters of the text for the next steps
raw_text = raw_text[:1000000]


# In[74]:


import os

if not os.path.exists("../datasets/raw_text.txt"):
    print("Speichere raw_text...")
    with open("../datasets/raw_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    print("Gespeichert!")
else:
    print("raw_text bereits gespeichert - wird nicht neu gespeichert.")


# ## Building a simple tokenizer
# 
# In this step, we develop a straightforward tokenizer to process the concatenated dataset:
# - The input string is split at spaces and specific punctuation marks that serve as word separators, including: , . : ; ? _ ! " ( )
# - This process results in a list of tokens. In this context, a token can be a complete word, a single character, or a sequence of special symbols (e.g., @#).
# ---
# 
# **Example**
# 
# ```This is a sample text: with numbers 123, special characters !@# and a few more. .```
# 
# The tokenizer identifies individual words like "text", but also preserves concatenated special characters such as "@#" as distinct tokens.
# 

# In[75]:


# the tokenizer function (basically just a split function)
def split_text(text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [token for token in tokens if token.strip()]
    return tokens


# In[76]:


# Example
example_text = "This is a sample text: with numbers 123, special characters !@# and a few more.  ."
result = split_text(example_text)
print(result)


# In[77]:


# Tokenizer on our dataset
split = split_text(raw_text)
print(f"Anzahl der Wörter: {len(split)}")
print(split)


# ## Converting Tokens into Token IDs
# This is a crucial preprocessing step for the neural network, as the model can only process numerical values.
# In the following cell, you can see a function and its output demonstrating how to map a token to an ID. The process is straightforward: the first token in the list is assigned ID 1, the second token is assigned ID 2, and so on.
# 
# This outputs the **vocabulary** (vocab), a list of all token ids mapped with the tokens of the dataset.

# In[78]:


all_words = set(split)
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# ## Building a whole tokenizer system
# The cell below combines these components into a Tokenizer class. This class takes text as input, splits it into tokens, and maps those tokens to IDs, as we saw in the cells above. Additionally, it includes a decoding function to convert token IDs back into strings. This decoding step is essential for us to translate the model's numerical output back into human-readable text.

# In[79]:


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# In[80]:


# Example on our raw_text
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(raw_text)
print(f"Input text: {raw_text}\n")
print(f"Token IDs: {ids}")

decoded_ids = tokenizer.decode(ids)
print(f"\nToken IDs decoded to text: {decoded_ids}")


# ## Add specials tokens
# In the next step, we integrate special tokens into our tokenizer to handle specific structural and vocabulary requirements:
# - <|unk|> (Unknown Token): This token represents words or symbols that were not present in the training vocabulary. Instead of throwing an error when encountering an unseen word during encoding, the system simply maps it to the <|unk|> ID
# - <|endoftext|> (End of Text Token): This serves as a signal to the model that the current sentence or phrase has ended.
# ---
# **Example**
# 
# With the vocabulary of our (reduced) dataset (see above) and the following input example:
# 
# ``Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.``
# 
# We get this encoded list of token IDs:
# 
# ``[19, 0, 19, 19, 19, 19, 19, 18, 19, 19, 19, 19, 19, 19, 19, 1]``
# 
# When we decode these token IDs, we get the following tokens in return:
# 
# ``<|unk|>, <|unk|> <|unk|> <|unk|> <|unk|> <|unk|> <|endoftext|> <|unk|> <|unk|> <|unk|> <|unk|> <|unk|> <|unk|> <|unk|>.``
# 
# This means that all of these words are unfortunaly unknown to our vocabulary. Only the punctuation marks ',' and '.' are recognized as known tokens.

# In[81]:


#adjusted tokenizer
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# In[82]:


# add the new special tokens to the vocabulary
all_tokens = sorted(list(set(split)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}


# In[83]:


# Test the new tokenizer with the updated vocabulary on an example input text
tokenizer = SimpleTokenizerV2(vocab)

# Example sentences
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(f"Example text: \n{text}\n")

# Get the token IDs for the example text
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}\n")

decoded_tokens = tokenizer.decode(token_ids)
print(f"Decoded text: \n{decoded_tokens}\n")


# ## Byte-pair-encoding
# 
# Our current tokenizer only recognizes exact words. By breaking unfamiliar terms into sub-tokens instead of simply labeling them as "unknown," we ensure the model can still process new words as long as their individual components are part of the vocabulary.
# 
# Instead of building this by our own, we use OpenAI’s tiktoken library with the cl100k_base encoding. Compared to the older gpt2 encoding (used as an example in the original chapters), it offers a larger vocabulary, a superior text compression and better support for special characters.
# 
# ---
# **Example**
# 
# The example below demonstrates the impact of Byte-Pair Encoding (BPE) compared to our SimpleTokenizer. Observe how the token count changes when spaces are removed.
# 
# Standard text:
# 
# ``Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.``
# 
# Compressed text without white spaces:
# 
# ``Hello,doyouliketea?<|endoftext|>Inthesunlitterracesofthepalace.``
# 
# - SimpleTokenizerV2:
#     - Standard text: Generates 16 tokens.
#     - Compressed text (no spaces): Generates only 6 tokens. (It fails to recognize individual words without spaces).
# 
# - Tiktoken (OpenAI):
#     - Standard text: Generates 19 tokens.
#     - Compressed text (no spaces): Generates 22 tokens. (It identifies sub-word units, maintaining the semantic meaning even without whitespace).

# In[84]:


tokenizer = SimpleTokenizerV2(vocab)

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.")

# Get the token IDs for the example text
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}\n")
print(f"Token Count: {len(token_ids)}")

decoded_tokens = tokenizer.decode(token_ids)
print(f"Decoded text: \n{decoded_tokens}\n")

# Decode tokens individually into a list without repr()
decoded_list = [tokenizer.decode([t_id]) for t_id in token_ids]

# Print the list
print("List of individual tokens:")
print(decoded_list)


# In[85]:


text = ( "Hello,doyouliketea?<|endoftext|>Inthesunlitterracesofthepalace." )

# Get the token IDs for the example text
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}\n")
print(f"Token Count: {len(token_ids)}")

decoded_tokens = tokenizer.decode(token_ids)
print(f"Decoded text: \n{decoded_tokens}\n")

# Decode tokens individually into a list without repr()
decoded_list = [tokenizer.decode([t_id]) for t_id in token_ids]

# Print the list
print("List of individual tokens:")
print(decoded_list)


# In[86]:


tokenizer = tiktoken.get_encoding("cl100k_base")

# Example text with the byte-pair-encoding
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(f"Token IDs: {token_ids}")
print(f"Token Count: {len(token_ids)}")

decoded_tokens = tokenizer.decode(token_ids)
print(f"\nDecoded Tokens: \n{decoded_tokens}\n")

# Decode tokens individually into a list without repr()
decoded_list = [tokenizer.decode([t_id]) for t_id in token_ids]

# Print the list
print("List of individual tokens:")
print(decoded_list)


# In[87]:


# Example text but without spaces to see the effect of the byte-pair-encoding
text = ( "Hello,doyouliketea?<|endoftext|>Inthesunlitterracesofthepalace." )
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(f"Token IDs: {token_ids}")
print(f"Token Count: {len(token_ids)}")

decoded_tokens = tokenizer.decode(token_ids)
print(f"\nDecoded Tokens: \n{decoded_tokens}\n")

# Decode tokens individually into a list without repr()
decoded_list = [tokenizer.decode([t_id]) for t_id in token_ids]

# Print the list
print("List of individual tokens:")
print(decoded_list)


# In[88]:


# Generate a comparison table for the two tokenizers on the two example texts

# Initialize tokenizers
tokenizer_v2 = SimpleTokenizerV2(vocab)
tokenizer_tik = tiktoken.get_encoding("cl100k_base")

# Define input strings
text_standard = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
text_compressed = "Hello,doyouliketea?<|endoftext|>Inthesunlitterracesofthepalace."

results = []

# Process strings
for label, text in [("Standard text (with spaces)", text_standard), ("Compressed text (no spaces)", text_compressed)]:
    # SimpleTokenizerV2
    ids_v2 = tokenizer_v2.encode(text)
    
    # Tiktoken (BPE)
    ids_tik = tokenizer_tik.encode(text, allowed_special={"<|endoftext|>"})
    
    results.append({
        "label": label,
        "v2_count": len(ids_v2),
        "tik_count": len(ids_tik)
    })

# Print comparison table
print(f"{'Text Version':<30} | {'SimpleV2 Count':<15} | {'Tiktoken Count':<15}")
print("-" * 60)
for res in results:
    print(f"{res['label']:<30} | {res['v2_count']:<15} | {res['tik_count']:<15}")


# ## Data Sampling with a Sliding Window
# 
# Once the text is tokenized, we need a mechanism to feed it into the neural network. Since a model learns to predict tokens sequentially, we train it to look at a sequence of previous words to predict the next one. To prevent the model from "cheating," it should only see the context preceding the target token.
# 
# To implement this, we use the Sliding Window approach:
# - **Input Sequence:** A fixed-length window of tokens from the text.
# - **Target Sequence:** The exact same window, but shifted by one position to the right.
# 
# ### DataLoader
# To efficiently feed our text into the model, we wrap the sliding window logic into a PyTorch Dataset and DataLoader. This allows us to handle large amounts of data, shuffle our sequences, and organize them into batches.
# 
# 
# **Key Parameters**
# 
# ``max_length``: The size of the "Context Window." It determines how many tokens the model looks at to predict the next one.
# 
# ``stride``: The distance the window moves for the next sample. A stride smaller than max_length creates overlapping sequences, providing more training data and helps the model see words in various positions.
# 
# ``batch_size``: The number of sequences processed simultaneously in one training step.
# 
# ``shuffle``: When True, it randomizes the order of the sequences to prevent the model from memorizing the order of the source text.
# 
# ---
# 
# **Example:**
# 
# The example below demonstrates how the DataLoader structures the data into Inputs (what the model sees) and Targets (what the model must predict). The targets are simply the inputs shifted by one position.
# 
# Setup:
# - ``batch_size=1``
# - ``max_length=4``
# - ``stride=1``
# 
# Input Text:
# ``Recepie: No-Bake Nut Cookies
# Ingredients: 1 c. firmly packed brown sugar, 1/2 c. evaporated milk, 1/``
# 
# DataLoader Output First Batch:
# ``[tensor([[  697,   346, 20898,    25]]), tensor([[  346, 20898,    25,  2360]])]``
# 
# Explanation:
# 
# ``tensor([[  697,   346, 20898,    25]])`` -> Input Tensor  
# (Corresponds to: ['Re', 'ce', 'pie', ':'] -> "Recepie:")
# 
# ``tensor([[  346, 20898,    25,  2360]])]`` -> Target Tensor  
# (Corresponds to: ['ce', 'pie', ':', ' No'] -> "cepie: No")

# In[89]:


from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# In[90]:


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# In[91]:


def decode_batch(batch, tokenizer):
    inputs, targets = batch
    
    # Extract IDs for the first sequence in the batch
    input_ids = inputs[0].tolist()
    target_ids = targets[0].tolist()
    
    # Decode the full text
    input_text = tokenizer.decode(input_ids)
    target_text = tokenizer.decode(target_ids)
    
    # Decode tokens individually to see the fragments
    input_tokens = [tokenizer.decode([t_id]) for t_id in input_ids]
    target_tokens = [tokenizer.decode([t_id]) for t_id in target_ids]

    print(f"Input batch (Full):   {input_text}")
    print(f"Input tokens (List):   {input_tokens}")
    print("-" * 30)
    print(f"Target batch (Full):  {target_text}")
    print(f"Target tokens (List):  {target_tokens}")


# In[92]:


# get the tokenized text
tokenizer = tiktoken.get_encoding("cl100k_base")
enc_text = tokenizer.encode(raw_text)
print(f"Raw text: {raw_text}")
print(f"\nEncoded text: {enc_text}")
print(f"\nLength of encoded text: {len(enc_text)}")


# In[93]:


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# Text verfication
decode_batch(first_batch, tokenizer)


# In[94]:


second_batch = next(data_iter)
print(second_batch)

# Text verfication
decode_batch(second_batch, tokenizer)


# In[95]:


#Quick Example how the dataloader works with batches
dataloader = create_dataloader_v1(raw_text, batch_size=3, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# ## Token embeddings
# 
# Token Embeddings transform discrete integer IDs into continuous vectors of fixed size. Instead of treating words as arbitrary numbers, we represent them in a high-dimensional space where the model can learn relationships between them.
# 
# An embedding layer is essentially a lookup table. Instead of performing expensive matrix multiplications with one-hot encoded vectors, it directly retrieves the corresponding row from a weight matrix. Over time, the model adjusts these vectors so that words used in similar contexts end up closer together in the vector space.
# 
# ### Understandings for ourself
# 
# **Why not just use IDs?**
# 
# While a tokenizer assigns a unique ID (integer) to every word, using these IDs directly for training is problematic:
# - Arbitrary Distance: Is token 500 "closer" in meaning to 501 than to 10? No. In integer form, the values are arbitrary and convey no semantic relationship.
# - Linear Scaling: A model would treat ID 1000 as "100 times more significant" than ID 10, which makes no sense for language.
# 
# Instead we are using embeddings.
# 
# **What are Embeddings?**
# 
# An Embedding replaces a single ID with a vector (a list of numbers).
# Instead of one number, we use multiple dimensions (e.g., 256, 768, or 1536) to describe a token. Each dimension can theoretically learn a different feature of the word—such as its grammatical role, sentiment, or relationship to other concepts. This allows the model to calculate similarity: words like "cat" and "dog" will eventually have similar vectors and sit close together in this high-dimensional space.
# 
# These Embeddings are combined in an big Embedding Matrix.
# 
# **Embedding Matrix**
# 
# The Embedding Matrix is essentially a giant lookup table.
# - Rows: The number of rows equals the Vocabulary Size (e.g., 100,277 for cl100k_base). This ensures every possible token has exactly one entry.
# - Columns: The number of columns is the Embedding Dimension (e.g., 768).
# 
# How it works: When you pass a Token ID to the layer, it simply retrieves the row at that index. ID 1 corresponds to Row 1. There is no calculation—just a fast lookup.
# 
# ---
# 
# **Example**
# 
# To visualize this, we use a tiny $6 \times 3$ Matrix. This means the model only knows 6 unique tokens, and each token is described by only 3 dimensions. At the start of training, these numbers are random and have no meaning—the model learns the "meaning" through the training.

# In[96]:


input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)


# In[97]:


# convert token ID 3 to its corresponding embedding vector
# Our Token ID is 3, so we look up the 3rd row of the embedding layer's weight matrix
print(embedding_layer(torch.tensor([3])))


# In[98]:


# Embedding every token
# This transformed matrix now serves as the numerical input for the  subsequent layers of the neural network during training.
print(embedding_layer(input_ids))


# ## Encoding word positions
# 
# At this stage, we have successfully transformed our Token IDs into dense vectors. However, there is a fundamental problem: **Token Embeddings are position-invariant.**
# 
# The embedded vectors of the sentences "The dog bites the man" and "The man bites the dog" are identical in both cases. Without additional information, a Transformer model would treat these sentences as a "bag of words," unaware of the order. To fix this, we use **Positional Embeddings**.
# 
# We create a second embedding matrix that doesn't store "word meanings," but rather "position meanings" (e.g., a vector for "Position 0", a vector for "Position 1").
# 
# To combine these two pieces of information, we simply add the Token Embedding vector and the Positional Embedding vector together. The resulting vector contains both the semantic meaning of the word and its location in the sequence.

# In[99]:


# The BytePair encoder hclk100k_base has a vocabulary size of 10,0277:
# Suppose we want to encode the input tokens into a 256-dimensional vector representation:
vocab_size = 100277
output_dim = 256

# 1.) Initialize the embedding layer with the specified vocabulary size and output dimension
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# In[100]:


# Extract the first batch of token IDs but with different parameters for the dataloader

max_length = 4
batch_size = 8

dataloader = create_dataloader_v1(
    raw_text, batch_size=batch_size, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)

inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("")
print(token_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
print(token_embeddings)


# In[101]:


# Extract the first batch of token IDs but with different parameters for the dataloader

max_length = 3
batch_size = 3

dataloader = create_dataloader_v1(
    raw_text, batch_size=batch_size, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)

inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("")
print(token_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
print(token_embeddings)


# In[102]:


# Create a positional embedding layer for the context length of the model
# The context length is the maximum number of tokens that the model can process in a single forward pass.
# Row 0 of the layer stores the vector for: "I am at the 1st position in the sequence."
# Row 1 stores the vector for: "I am at the 2nd position in the sequence"
# and so on...

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# uncomment & execute the following line to see how the embedding layer weights look like
print(pos_embedding_layer.weight)


# In[103]:


# Creates a simple tensor of indices: [0, 1, 2, 3]. These are positional addresses.
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
print(pos_embeddings)


# In[104]:


# To create now the final input embeddings used in an LLM, we simply add the token and the positional embeddings:
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
print(input_embeddings)


# ## Short Summary
# 
# We have successfully transformed raw text into a input embedding, a format that a Large Language Model can "understand" and process.
# 
# To reach the final input, we combined two distinct layers:
# 
# - Token Embeddings: Convert Token IDs into 256-dimensional vectors to capture semantic meaning.
# - Positional Embeddings: Add unique vectors for each position to capture word order.
# 
# The resulting input_embeddings tensor is now ready for the neural network.
