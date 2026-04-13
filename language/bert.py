import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BertConfig:
	vocab_size: int = 30522
	hidden_size: int = 256
	num_hidden_layers: int = 4
	num_attention_heads: int = 8
	intermediate_size: int = 1024
	hidden_dropout_prob: float = 0.1
	attention_probs_dropout_prob: float = 0.1
	max_position_embeddings: int = 512
	type_vocab_size: int = 2
	layer_norm_eps: float = 1e-12


class BertEmbeddings(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(
			config.max_position_embeddings,
			config.hidden_size,
		)
		self.token_type_embeddings = nn.Embedding(
			config.type_vocab_size,
			config.hidden_size,
		)

		self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(
		self,
		input_ids: torch.Tensor,
		token_type_ids: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		batch_size, seq_len = input_ids.shape
		device = input_ids.device

		if token_type_ids is None:
			token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

		position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

		words = self.word_embeddings(input_ids)
		positions = self.position_embeddings(position_ids)
		token_types = self.token_type_embeddings(token_type_ids)

		embeddings = words + positions + token_types
		embeddings = self.layer_norm(embeddings)
		return self.dropout(embeddings)


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError("hidden_size must be divisible by num_attention_heads")

		self.num_heads = config.num_attention_heads
		self.head_dim = config.hidden_size // config.num_attention_heads
		self.all_head_dim = self.num_heads * self.head_dim

		self.query = nn.Linear(config.hidden_size, self.all_head_dim)
		self.key = nn.Linear(config.hidden_size, self.all_head_dim)
		self.value = nn.Linear(config.hidden_size, self.all_head_dim)
		self.out = nn.Linear(config.hidden_size, config.hidden_size)

		self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
		self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)

	def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
		return x.transpose(1, 2)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		query = self._reshape_to_heads(self.query(hidden_states))
		key = self._reshape_to_heads(self.key(hidden_states))
		value = self._reshape_to_heads(self.value(hidden_states))

		attn_scores = torch.matmul(query, key.transpose(-1, -2))
		attn_scores = attn_scores / math.sqrt(self.head_dim)

		if attention_mask is not None:
			attn_scores = attn_scores + attention_mask

		attn_probs = torch.softmax(attn_scores, dim=-1)
		attn_probs = self.attn_dropout(attn_probs)

		context = torch.matmul(attn_probs, value)
		context = context.transpose(1, 2).contiguous()
		batch_size, seq_len, _, _ = context.shape
		context = context.view(batch_size, seq_len, self.all_head_dim)

		output = self.out(context)
		return self.proj_dropout(output)


class FeedForward(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
		self.act = nn.GELU()
		self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.dense_in(x)
		x = self.act(x)
		x = self.dense_out(x)
		return self.dropout(x)


class BertLayer(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.self_attn = MultiHeadSelfAttention(config)
		self.attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

		self.ffn = FeedForward(config)
		self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		attn_output = self.self_attn(hidden_states, attention_mask=attention_mask)
		hidden_states = self.attn_layer_norm(hidden_states + attn_output)

		ffn_output = self.ffn(hidden_states)
		hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
		return hidden_states


class BertEncoder(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		for layer in self.layers:
			hidden_states = layer(hidden_states, attention_mask=attention_mask)
		return hidden_states


class BertPooler(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		# BERT uses the first token ([CLS]) as the pooled representation.
		first_token = hidden_states[:, 0]
		return self.activation(self.dense(first_token))


class BertModel(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.config = config
		self.embeddings = BertEmbeddings(config)
		self.encoder = BertEncoder(config)
		self.pooler = BertPooler(config)

	@staticmethod
	def _create_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
		# Convert mask [B, S] with 1/0 values into additive mask [B, 1, 1, S].
		extended = attention_mask[:, None, None, :].to(dtype=torch.float32)
		return (1.0 - extended) * -1e4

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)

		extended_attention_mask = self._create_attention_mask(attention_mask)

		embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
		sequence_output = self.encoder(embedding_output, attention_mask=extended_attention_mask)
		pooled_output = self.pooler(sequence_output)
		return sequence_output, pooled_output


class BertForMaskedLM(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.bert = BertModel(config)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		self.bias = nn.Parameter(torch.zeros(config.vocab_size))
		self.lm_head.bias = self.bias

		# Tie decoder weights to token embeddings (standard BERT trick).
		self.lm_head.weight = self.bert.embeddings.word_embeddings.weight

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		sequence_output, _ = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)
		logits = self.lm_head(sequence_output)

		loss = None
		if labels is not None:
			loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
			loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

		return logits, loss


if __name__ == "__main__":
	# Tiny sanity check with random token ids.
	cfg = BertConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
	model = BertForMaskedLM(cfg)

	batch_size, seq_len = 2, 16
	inputs = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
	mask = torch.ones((batch_size, seq_len), dtype=torch.long)
	labels = inputs.clone()
	labels[:, :4] = -100

	logits, loss = model(input_ids=inputs, attention_mask=mask, labels=labels)
	print(f"logits shape: {tuple(logits.shape)}")
	print(f"loss: {None if loss is None else float(loss):.4f}")
