import torch
from torch import nn
from random import randint


with open('chr_en_data/train.chr') as f: chr_texts = [l.strip() for l in f.readlines()]
with open('chr_en_data/train.en')  as f: en_texts  = [l.strip() for l in f.readlines()]

chr_vocabulary = list(set(''.join(chr_texts))) + ['<START>', '<END>']
en_vocabulary  = list(set(''.join(en_texts)))  + ['<START>', '<END>']
print(en_vocabulary)
chr_encoder = {v : i for i, v in enumerate(chr_vocabulary)}
chr_decoder = {i : v for i, v in enumerate(chr_vocabulary)}

en_encoder = {v : i for i, v in enumerate(en_vocabulary)}
en_decoder = {i : v for i, v in enumerate(en_vocabulary)}


max_length = 0
for i, text in enumerate(chr_texts + en_texts): 
	max_length = max(max_length, len(text) + 2)
	
chr_encoded = torch.full(size=(len(chr_texts), max_length), fill_value=-1)
en_encoded  = torch.full(size=(len(en_texts), max_length),  fill_value=-1)


for i, v in enumerate(chr_texts):
	chr_encoded[i, 0], en_encoded[i, 0] = chr_encoder['<START>'], en_encoder['<START>']

	for j, c in enumerate(v): chr_encoded[i, j+1] = chr_encoder[c]
	chr_encoded[i, j+2] = chr_encoder['<END>']

	for j, c in enumerate(en_texts[i]): en_encoded[i, j+1] = en_encoder[c]
	en_encoded[i, j+2] = en_encoder['<END>']

CONTEXT_WINDOW = max_length
MASK_PERCENTAGE = 0.15
EN_VOCABULARY_SIZE  = len(en_vocabulary)
CHR_VOCABULARY_SIZE = len(chr_vocabulary) 
MASKS_PER_SEQUENCE = int(CONTEXT_WINDOW * MASK_PERCENTAGE)
EMBEDDING_SIZE = 64
NUM_HEADS      = 4
HEAD_SIZE      = EMBEDDING_SIZE // NUM_HEADS
BATCH_SIZE     = 64
NUM_ENCODER_BLOCKS = 4
DROPOUT_RATE = 0.1
EPOCHS = 1000

def get_batch():
	
	indices = torch.randint(low=0, high=int(0.9*len(chr_texts)), size=(BATCH_SIZE,))
	tokens_to_predict = torch.randint(low=1, high=CONTEXT_WINDOW, size=(BATCH_SIZE,))

	X_1 = torch.full(size=(BATCH_SIZE, max_length), fill_value=-1) # encoder input
	X_2 = torch.full(size=(BATCH_SIZE, max_length), fill_value=-1) # decoder input
	Y   = torch.zeros(size=(BATCH_SIZE,), dtype=torch.long)

	for i, ix in enumerate(indices):
		j = tokens_to_predict[i]
		if en_encoded[ix][j] == -1:
			max_index = torch.where(en_encoded[ix] == en_encoder['<END>'])[0].item()
			j = randint(1, max_index)
		X_1[i] = chr_encoded[ix]
		X_2[i] = en_encoded[ix, :j]
		Y[i]   = en_encoded[ix, j]

	return X_1, X_2, Y
		

class FeedForward(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
			nn.Dropout(DROPOUT_RATE),
			nn.ReLU(),
			nn.Dropout(DROPOUT_RATE),
			nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
		)

	def forward(self, data):
		return self.network(data)
	
class AttentionHead(nn.Module):
	def __init__(self):
		super().__init__()
		self.key = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.query = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.value = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, data):		
		key = self.key(data)
		query = self.query(data)
		value = self.value(data)

		mat_mul = query @ key.transpose(-2, -1) # Dot Product
		scaled_mat_mul = self.softmax(mat_mul * (1 / torch.sqrt(HEAD_SIZE)))
		return scaled_mat_mul @ value


class MultiHeadedAttention(nn.Module):
	def __init__(self):
		super().__init__()
		self.attention_heads = nn.ModuleList([AttentionHead() for i in range(NUM_HEADS)])
		self.linear = nn.Linear(HEAD_SIZE * NUM_HEADS, EMBEDDING_SIZE)
		self.dropout = nn.Dropout(DROPOUT_RATE)

	def forward(self, data):
		head_outputs = [head(data) for head in self.attention_heads]
		concatenated_outputs = torch.cat(head_outputs, dim=-1)
		projected_outputs = self.linear(concatenated_outputs)
		return self.dropout(projected_outputs)

class EncoderBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.attention = MultiHeadedAttention()
		self.layer_norm_1 = nn.LayerNorm(EMBEDDING_SIZE)
		self.feed_forward = FeedForward()
		self.layer_norm_2 = nn.LayerNorm(EMBEDDING_SIZE)

	def forward(self, data):
		attention_outputs = self.attention(data)
		normalised_data_1 = self.layer_norm_1(data + attention_outputs)
		normalised_data_2 = self.layer_norm_2(self.feed_forward(normalised_data_1) + normalised_data_1)

		return normalised_data_2

