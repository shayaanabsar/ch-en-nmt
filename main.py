import torch
from torch import nn
import os

# --- Data Loading and Preprocessing ---
# Define file paths
CHAR_TRAIN_FILE = 'chr_en_data/train.chr'
EN_TRAIN_FILE = 'chr_en_data/train.en'

# Basic file existence checks
if not os.path.exists(CHAR_TRAIN_FILE):
	raise FileNotFoundError(f"Training file not found: {CHAR_TRAIN_FILE}")
if not os.path.exists(EN_TRAIN_FILE):
	raise FileNotFoundError(f"Training file not found: {EN_TRAIN_FILE}")

try:
	with open(CHAR_TRAIN_FILE, 'r', encoding='utf-8') as f:
		chr_texts = [line.strip() for line in f.readlines()]
	with open(EN_TRAIN_FILE, 'r', encoding='utf-8') as f:
		en_texts  = [line.strip() for line in f.readlines()]
except Exception as e:
	raise IOError(f"Error reading data files: {e}")

if not chr_texts or not en_texts or len(chr_texts) != len(en_texts):
	raise ValueError("Data loading error: files empty or line counts mismatch.")

# --- Vocabulary Creation ---
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

chr_vocabulary = [PAD_TOKEN, START_TOKEN, END_TOKEN] + list(set(''.join(chr_texts)))
en_vocabulary  = [PAD_TOKEN, START_TOKEN, END_TOKEN] + list(set(''.join(en_texts)))

chr_encoder = {v : i for i, v in enumerate(chr_vocabulary)}
chr_decoder = {i : v for i, v in enumerate(chr_vocabulary)}

en_encoder = {v : i for i, v in enumerate(en_vocabulary)}
en_decoder = {i : v for i, v in enumerate(en_vocabulary)}


EN_VOCABULARY_SIZE  = len(en_vocabulary)
CHR_VOCABULARY_SIZE = len(chr_vocabulary)

# --- Determine Max Sequence Length ---
max_length = 0
for text in chr_texts + en_texts:
	max_length = max(max_length, len(text) + 2) # +2 for <START> and <END>

# --- Encode All Data ---
chr_encoded = torch.full(size=(len(chr_texts), max_length), fill_value=chr_encoder[PAD_TOKEN])
en_encoded  = torch.full(size=(len(en_texts), max_length),  fill_value=en_encoder[PAD_TOKEN])

for i, source_text in enumerate(chr_texts):
	target_text = en_texts[i]

	# Source language encoding
	chr_encoded[i, 0] = chr_encoder[START_TOKEN]
	chars_to_encode_src = min(len(source_text), max_length - 2)
	for j in range(chars_to_encode_src):
		char = source_text[j]
		chr_encoded[i, j+1] = chr_encoder[char]
	chr_encoded[i, 1 + chars_to_encode_src] = chr_encoder[END_TOKEN]

	# Target language encoding
	en_encoded[i, 0] = en_encoder[START_TOKEN]
	chars_to_encode_tgt = min(len(target_text), max_length - 2)
	for j in range(chars_to_encode_tgt):
		char = target_text[j]
		en_encoded[i, j+1] = en_encoder[char]
	en_encoded[i, 1 + chars_to_encode_tgt] = en_encoder[END_TOKEN]

# --- Hyperparameters and Configuration ---
CONTEXT_WINDOW = max_length
EMBEDDING_SIZE = 64
NUM_HEADS      = 4
HEAD_SIZE      = EMBEDDING_SIZE // NUM_HEADS
BATCH_SIZE     = 8
NUM_ENCODER_BLOCKS = 4
NUM_DECODER_BLOCKS = 4
DROPOUT_RATE = 0.1
EPOCHS = 1000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data Batching Utility ---
def get_batch():
	num_training_samples = int(0.9 * len(chr_texts))
	if num_training_samples == 0: raise ValueError("Not enough training samples.")

	indices = torch.randint(low=0, high=num_training_samples, size=(BATCH_SIZE,)).to(DEVICE)

	encoder_input_batch = torch.full(size=(BATCH_SIZE, max_length), fill_value=chr_encoder[PAD_TOKEN], dtype=torch.long, device=DEVICE)
	decoder_input_batch = torch.full(size=(BATCH_SIZE, max_length), fill_value=en_encoder[PAD_TOKEN], dtype=torch.long, device=DEVICE)
	target_output_batch = torch.full(size=(BATCH_SIZE, max_length), fill_value=en_encoder[PAD_TOKEN], dtype=torch.long, device=DEVICE)

	for i, idx in enumerate(indices):
		src_seq = chr_encoded[idx].to(DEVICE)
		tgt_seq = en_encoded[idx].to(DEVICE)

		decoder_input = tgt_seq[:-1]
		target_output = tgt_seq[1:]

		encoder_input_batch[i, :len(src_seq)]           = src_seq
		decoder_input_batch[i, :len(decoder_input)]     = decoder_input
		target_output_batch[i, :len(target_output)]     = target_output

	return encoder_input_batch, decoder_input_batch, target_output_batch

# --- Model Architecture ---
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
	def forward(self, data): return self.network(data)
	
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

		mat_mul = Q @ K.transpose(-2, -1) # Shape (B, T, T)
		scaled_mat_mul = mat_mul * (1 / (HEAD_SIZE ** 0.5))

		if self.masked:
			scaled_mat_mul = scaled_mat_mul.masked_fill(self.tril == 0, float('-inf'))
			
		attention_weights = self.softmax(scaled_mat_mul)
		return attention_weights @ V

class MultiHeadedAttention(nn.Module):
	def __init__(self, masked=False):
		super().__init__()
		self.attention_heads = nn.ModuleList([AttentionHead(masked=masked) for _ in range(NUM_HEADS)])
		self.linear = nn.Linear(HEAD_SIZE * NUM_HEADS, EMBEDDING_SIZE)
		self.dropout = nn.Dropout(DROPOUT_RATE)

	def forward(self, query, key=None, value=None):
		if key is None: key = query
		if value is None: value = query
		head_outputs = [head(query, key, value) for head in self.attention_heads]
		concatenated_outputs = torch.cat(head_outputs, dim=-1)
		projected_outputs = self.linear(concatenated_outputs)
		return self.dropout(projected_outputs)

class EncoderBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.self_attention = MultiHeadedAttention(masked=False)
		self.layer_norm_1 = nn.LayerNorm(EMBEDDING_SIZE)
		self.feed_forward = FeedForward()
		self.layer_norm_2 = nn.LayerNorm(EMBEDDING_SIZE)

	def forward(self, data):
		attention_outputs = self.self_attention(data)
		normalised_data_1 = self.layer_norm_1(data + attention_outputs)
		ff_outputs = self.feed_forward(normalised_data_1)
		normalised_data_2 = self.layer_norm_2(ff_outputs + normalised_data_1)
		return normalised_data_2

class DecoderBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.masked_self_attention = MultiHeadedAttention(masked=True)
		self.layer_norm_1 = nn.LayerNorm(EMBEDDING_SIZE)
		self.cross_attention  = MultiHeadedAttention(masked=False)
		self.layer_norm_2 = nn.LayerNorm(EMBEDDING_SIZE)
		self.feed_forward = FeedForward()
		self.layer_norm_3 = nn.LayerNorm(EMBEDDING_SIZE)

	def forward(self, data):
		decoder_input, encoder_output = data
		masked_attn_outputs = self.masked_self_attention(decoder_input)
		normalised_masked_attn = self.layer_norm_1(masked_attn_outputs + decoder_input)
		cross_attn_outputs = self.cross_attention(query=normalised_masked_attn, key=encoder_output, value=encoder_output)
		normalised_cross_attn = self.layer_norm_2(cross_attn_outputs + normalised_masked_attn)
		ff_outputs = self.feed_forward(normalised_cross_attn)
		normalised_ff_outputs = self.layer_norm_3(ff_outputs + normalised_cross_attn)
		return normalised_ff_outputs
	
class Transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder_tok_embeddings = nn.Embedding(CHR_VOCABULARY_SIZE, EMBEDDING_SIZE)
		self.encoder_pos_embeddings = nn.Embedding(CONTEXT_WINDOW, EMBEDDING_SIZE)

		self.decoder_tok_embeddings = nn.Embedding(EN_VOCABULARY_SIZE, EMBEDDING_SIZE)
		self.decoder_pos_embeddings = nn.Embedding(CONTEXT_WINDOW, EMBEDDING_SIZE)

		self.encoders = nn.ModuleList([EncoderBlock() for _ in range(NUM_ENCODER_BLOCKS)])
		self.decoders = nn.ModuleList([DecoderBlock() for _ in range(NUM_DECODER_BLOCKS)])

		self.output_linear = nn.Linear(EMBEDDING_SIZE, EN_VOCABULARY_SIZE)

	def forward(self, data):
		encoder_input, decoder_input = data
		B, T_enc = encoder_input.shape
		token_embeddings_enc = self.encoder_tok_embeddings(encoder_input)
		positional_embeddings_enc = self.encoder_pos_embeddings(torch.arange(T_enc, device=encoder_input.device)).unsqueeze(0).expand(B, T_enc, -1)
		encoder_embedding = token_embeddings_enc + positional_embeddings_enc

		for block in self.encoders: encoder_embedding = block(encoder_embedding)

		B, T_dec = decoder_input.shape
		token_embeddings_dec = self.decoder_tok_embeddings(decoder_input)
		positional_embeddings_dec = self.decoder_pos_embeddings(torch.arange(T_dec, device=decoder_input.device)).unsqueeze(0).expand(B, T_dec, -1)
		decoder_embedding = token_embeddings_dec + positional_embeddings_dec

		for block in self.decoders: decoder_embedding = block((decoder_embedding, encoder_embedding))

		logits = self.output_linear(decoder_embedding)
		return logits
	
	def decode_output(self, decoder_sequence_tensor):
		translated_tokens = []
		sequence_list = decoder_sequence_tensor.squeeze(0).tolist()

		for token_id in sequence_list:
			if token_id == en_encoder[START_TOKEN]: continue
			elif token_id == en_encoder[END_TOKEN]: break
			elif token_id == en_encoder[PAD_TOKEN]: continue
			else:
				if token_id in en_decoder:
					translated_tokens.append(en_decoder[token_id])
				else:
					print(f"Warning: Unknown token ID {token_id} in decoder output.")
					translated_tokens.append('[UNK]') 
		return ''.join(translated_tokens)

	def translate(self, beam_width):
		self.eval() 
		pass
	
	
model = Transformer().to(DEVICE)

print("\n--- Example Translation ---")
#try:
#	test_text = 'ᎾᏍᎩᏃ ᎯᎠ ᏣᏂ ᎤᏄᏪ ᎨᎻᎵ ᎤᏍᏘᏰᏅᎯ, ᎦᏃᏥᏃ ᎤᏓᏠᏍᏕᎢ, ᎤᎵᏍᏓᏴᏗᏃ ᎥᎴ ᎨᏎ ᎢᎾᎨᏃ ᎡᎯ ᏩᏚᎵᏏ.'
#	translated_output = model.translate(test_text)
#	print(f"Input:    '{test_text}'")
#	print(f"Output:   '{translated_output}'")
#except Exception as e:
#	print(f"Error during translation: {e}")


loss_fn = nn.CrossEntropyLoss(ignore_index=en_encoder[PAD_TOKEN])
optim   = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for i in range(EPOCHS):
	encoder_input_batch, decoder_input_batch, target_output_batch = get_batch()
	logits = model((encoder_input_batch, decoder_input_batch))
	B, T, V = logits.shape # This line will now work correctly
	loss   = loss_fn(logits.view(B*T, V), target_output_batch.view(B*T))
	optim.zero_grad()
	loss.backward()
	optim.step()
	if i % 10 == 0: print(f'Loss at epoch {i} = {loss.item():.4f}')