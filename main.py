import torch
from torch import nn
from random import randint


with open('chr_en_data/train.chr') as f: chr_texts = [l.strip() for l in f.readlines()]
with open('chr_en_data/train.en')  as f: en_texts  = [l.strip() for l in f.readlines()]

chr_vocabulary = ['<PAD>', '<START>', '<END>'] + list(set(''.join(chr_texts)))
en_vocabulary  = ['<PAD>', '<START>', '<END>'] + list(set(''.join(en_texts)))

chr_encoder = {v : i for i, v in enumerate(chr_vocabulary)}
chr_decoder = {i : v for i, v in enumerate(chr_vocabulary)}

en_encoder = {v : i for i, v in enumerate(en_vocabulary)}
en_decoder = {i : v for i, v in enumerate(en_vocabulary)}


max_length = 0
for i, text in enumerate(chr_texts + en_texts): 
	max_length = max(max_length, len(text) + 2)
	
chr_encoded = torch.full(size=(len(chr_texts), max_length), fill_value=chr_encoder['<PAD>'])
en_encoded  = torch.full(size=(len(en_texts), max_length),  fill_value=en_encoder['<PAD>'])


for i, v in enumerate(chr_texts):
	chr_encoded[i, 0], en_encoded[i, 0] = chr_encoder['<START>'], en_encoder['<START>']

	for j, c in enumerate(v): chr_encoded[i, j+1] = chr_encoder[c]
	chr_encoded[i, j+2] = chr_encoder['<END>']

	for j, c in enumerate(en_texts[i]): en_encoded[i, j+1] = en_encoder[c]
	en_encoded[i, j+2] = en_encoder['<END>']

CONTEXT_WINDOW = max_length
EN_VOCABULARY_SIZE  = len(en_vocabulary)
CHR_VOCABULARY_SIZE = len(chr_vocabulary) 
EMBEDDING_SIZE = 64
NUM_HEADS      = 4
HEAD_SIZE      = EMBEDDING_SIZE // NUM_HEADS
BATCH_SIZE     = 8
NUM_ENCODER_BLOCKS = 4
NUM_DECODER_BLOCKS = 4
DROPOUT_RATE = 0.1
EPOCHS = 1000

def get_batch():
	
	indices = torch.randint(low=0, high=int(0.9*len(chr_texts)), size=(BATCH_SIZE,))

	X_1 = torch.full(size=(BATCH_SIZE, max_length), fill_value=chr_encoder['<PAD>']) # encoder input
	X_2 = torch.full(size=(BATCH_SIZE, max_length), fill_value=en_encoder['<PAD>']) # decoder input
	Y   = torch.zeros(size=(BATCH_SIZE, max_length), dtype=torch.long)

	for i, ix in enumerate(indices):
		src = chr_encoded[ix]
		tgt = en_encoded[ix]

		decoder_input = tgt[:-1]
		target_output = tgt[1:]

		X_1[i, :len(src)]        = src
		X_2[i, :len(decoder_input)] = decoder_input
		Y[i, :len(target_output)]   = target_output

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
	def __init__(self, masked=False):
		super().__init__()
		self.key = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.query = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.value = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_WINDOW, CONTEXT_WINDOW)))
		self.softmax = nn.Softmax(dim=-1)
		self.masked = masked

	def forward(self, query, key, value):		
		Q = self.query(query)
		K = self.key(key)
		V = self.value(value)

		mat_mul = Q @ K.transpose(-2, -1) # Dot Product
		scaled_mat_mul = mat_mul * (1 / (HEAD_SIZE ** 0.5))

		if self.masked:
			scaled_mat_mul = scaled_mat_mul.masked_fill(self.tril == 0, float('-inf'))
		attention_weights = self.softmax(scaled_mat_mul)
		return attention_weights @ V


class MultiHeadedAttention(nn.Module):
	def __init__(self, masked=False):
		super().__init__()
		self.attention_heads = nn.ModuleList([AttentionHead(masked=masked) for i in range(NUM_HEADS)])
		self.linear = nn.Linear(HEAD_SIZE * NUM_HEADS, EMBEDDING_SIZE)
		self.dropout = nn.Dropout(DROPOUT_RATE)

	def forward(self, query, key=None, value=None):
		if key is None : key=query
		if value is None: value=query
		head_outputs = [head(query, key, value) for head in self.attention_heads]
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

class DecoderBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.masked_attention = MultiHeadedAttention(masked=True)
		self.layer_norm_1 = nn.LayerNorm(EMBEDDING_SIZE)
		self.cross_attention  = MultiHeadedAttention()
		self.layer_norm_2 = nn.LayerNorm(EMBEDDING_SIZE)
		self.feed_forward = FeedForward()
		self.layer_norm_3 = nn.LayerNorm(EMBEDDING_SIZE)

	def forward(self, data):
		decoder_input, encoder_output = data
		masked_attention = self.masked_attention(decoder_input)
		normalised_masked_attention = self.layer_norm_1(masked_attention + decoder_input)
		cross_attention = self.cross_attention(query=normalised_masked_attention, key=encoder_output, value=encoder_output)
		normalised_cross_attn = self.layer_norm_2(cross_attention + normalised_masked_attention)
		projected = self.feed_forward(normalised_cross_attn)
		normalised_projected = self.layer_norm_3(projected + normalised_cross_attn)

		return normalised_projected
	

class Transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder_pos_embeddings = nn.Embedding(num_embeddings=CONTEXT_WINDOW, embedding_dim=EMBEDDING_SIZE)
		self.encoder_tok_embeddings = nn.Embedding(num_embeddings=CHR_VOCABULARY_SIZE, embedding_dim=EMBEDDING_SIZE)

		self.decoder_pos_embeddings = nn.Embedding(num_embeddings=CONTEXT_WINDOW, embedding_dim=EMBEDDING_SIZE)
		self.decoder_tok_embeddings = nn.Embedding(num_embeddings=EN_VOCABULARY_SIZE, embedding_dim=EMBEDDING_SIZE)

		self.encoders = nn.ModuleList([EncoderBlock() for i in range(NUM_ENCODER_BLOCKS)])
		self.decoders = nn.ModuleList([DecoderBlock() for i in range(NUM_DECODER_BLOCKS)])

		self.linear = nn.Linear(EMBEDDING_SIZE, EN_VOCABULARY_SIZE)

	
	def forward(self, data):
		encoder_input, decoder_input = data

		B, T = encoder_input.shape

		token_embeddings = self.encoder_tok_embeddings(encoder_input)
		positional_embeddings = self.encoder_pos_embeddings(torch.arange(T, device=encoder_input.device)).unsqueeze(0).expand(B, T, -1)
		encoder_embedding = token_embeddings + positional_embeddings

		for block in self.encoders: encoder_embedding = block(encoder_embedding)

		B, T = decoder_input.shape

		token_embeddings = self.decoder_tok_embeddings(decoder_input)
		positional_embeddings = self.decoder_pos_embeddings(torch.arange(T, device=decoder_input.device)).unsqueeze(0).expand(B, T, -1)
		decoder_embedding = token_embeddings + positional_embeddings

		for block in self.decoders: decoder_embedding = block((decoder_embedding, encoder_embedding))

		logits = self.linear(decoder_embedding)

		return logits
	

model = Transformer()

loss_fn = nn.CrossEntropyLoss(ignore_index=en_encoder['<PAD>'])
optim   = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for i in range(EPOCHS):
	X_1, X_2, Y = get_batch()

	logits = model((X_1, X_2))
	B, T, V = logits.shape
	loss   = loss_fn(logits.view(B*T, V), Y.view(B*T))
	

	optim.zero_grad()
	loss.backward()
	optim.step()

	if i % 10 == 0: print(f'Loss at epoch {i} = {loss.item():.4f}')
