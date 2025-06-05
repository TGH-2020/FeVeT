import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
import torch.nn.functional as F

class CognateS2S(nn.Module):
    def __init__(
        self,
        char2idx: dict,
        feat_dim: int = 39, # Number of features in SoundVectors library; increase if you add features
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        max_target_len: int = 20 # Maximum length for autoregressive decoding
        ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(char2idx) + 1
        self.max_target_len = max_target_len  
        self.eos_token_id = char2idx["[SEP]"]
        self.pad_token_id = char2idx["[PAD]"]

        # Language embedding layer
        self.lang_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.pad_token_id)

        # Feature vector projection
        self.feature_proj = nn.Linear(feat_dim, hidden_dim)
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim) * 0.01)

        # Transformer encoder
        self.encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = TransformerEncoder(self.encoder_layers, num_layers=num_encoder_layers)
        
        # Transformer decoder
        self.decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Character embedding layer for IPA tokens
        self.char_embedding = nn.Embedding(self.vocab_size, hidden_dim)

        # Project masked language embedding to decoder space
        self.target_lang_embed = nn.Embedding(self.vocab_size, hidden_dim)

        # [SOS] token (learnable)
        self.sos_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)

        # Output layer
        self.output_proj = PhonemeDecoder(hidden_dim, feat_dim*5, self.vocab_size)

    def forward(self, langs, inputs, target_langs, target, target_featvecs):
        """
        Args:
            langs: Tensor of shape (batch_size, batch_num_rows, batch_seq_len) containing language IDs.
            inputs: Tensor of shape (batch_size, batch_num_rows, batch_seq_len, feat_dim) containing feature vectors.
            target_langs: Tensor of shape (batch_size,) containing target language IDs.
            target: Tensor of shape (batch_size, target_len) containing ground truth IPA IDs (for teacher forcing).
        Returns:
            feat_logits: Tensor of shape (batch_size, target_len, feat_dim*5) containing feature vector predictions.
            ipa_logits: Tensor of shape (batch_size, target_len, vocab_size) containing IPA token predictions.

        """

        # Encode inputs
        encoder_output = self._encode(langs, inputs) # (batch_size, batch_num_rows * batch_seq_len, hidden_dim)
        
        # Embed target language
        target_lang_embed = self.target_lang_embed(target_langs)  # (batch_size, 1, hidden_dim)

        # Get target mask
        target_padding_mask = target != 0
        # Shift padding mask
        target_padding_mask = torch.cat([torch.ones(target.size(0), 1, dtype=torch.bool, device=target.device), target_padding_mask], dim=1)[:, :-1]

        # Teacher forcing: [SOS] + target (shifted)
        sos_embed = self.sos_embed.expand(langs.size(0), 1, -1)  # (batch_size, 1, hidden_dim)
        char_embeds = target_featvecs[:, :-1]  # (batch_size, target_len-1, feat_dim)
        char_embeds = self.feature_proj(char_embeds) # (batch_size, target_len-1, hidden_dim)

        decoder_input = torch.cat([sos_embed, char_embeds], dim=1)  # (batch_size, target_len, hidden_dim)
        # Add target language embedding to each position and mask
        decoder_input = decoder_input + target_lang_embed # (batch_size, target_len, hidden_dim)
        decoder_input = decoder_input * target_padding_mask.unsqueeze(-1)

        # Decode
        feat_logits, ipa_logits = self._decode_step(decoder_input, encoder_output)
        return feat_logits, ipa_logits

    def _encode(self, langs, inputs):
        """
        Encodes the input sequences.
        Args:
            langs: Tensor of shape (batch_size, batch_num_rows, batch_seq_len) containing language IDs for each position.
            inputs: Tensor of shape (batch_size, batch_num_rows, batch_seq_len, feat_dim) containing feature vectors.
        Returns:
            encoder_output: (batch_size, batch_num_rows * batch_seq_len, hidden_dim)
        """
        batch_size, num_rows, seq_len, _ = inputs.shape

        padding_mask = langs != 0 # (batch_size, num_rows, seq_len)

        # Language embeddings
        lang_embeds = self.lang_embedding(langs)  # (batch_size, num_rows, seq_len, hidden_dim)

        # Project feature vectors to hidden_dim and add language embeddings to each position
        inputs = self.feature_proj(inputs)  # (batch_size, num_rows, seq_len, hidden_dim)
        inputs = inputs + lang_embeds

        # Add positional encoding and mask padding
        inputs = inputs + self.positional_encoding[:seq_len].unsqueeze(0).unsqueeze(0)
        inputs = inputs * padding_mask.unsqueeze(-1) # (batch_size, num_rows, seq_len, hidden_dim)

        # Encoder over each row    
        encoder_output = inputs
        encoder_output = encoder_output.reshape(batch_size*num_rows, seq_len, encoder_output.size(-1))
        encoder_output = self.encoder(encoder_output)
        
        encoder_output = encoder_output.reshape(batch_size, num_rows, seq_len, encoder_output.size(-1))
        encoder_output = encoder_output.reshape(batch_size, num_rows * seq_len, encoder_output.size(-1))
           
        return encoder_output

    def _decode_step(self, decoder_input, encoder_output):
        """
        Performs a single decoding step.
        Args:
            decoder_input: (batch_size, target_len, hidden_dim)
            encoder_output: (seq_len, batch_size, hidden_dim)
        Returns:
            feat_logits: (batch_size, target_len, feat_dim*5)
            ipa_logits: (batch_size, target_len, vocab_size)
        """
        seq_len = decoder_input.size(1)

        # Add positional encoding
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
        decoder_input += pos_enc

        # Causal mask
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input.device), 
            diagonal=1
        ).bool()

        # Decoder step
        decoder_output = self.decoder(
            decoder_input,  # (batch_size, target_len, hidden_dim)
            encoder_output,  # (batch_size, seq_len, hidden_dim)
            tgt_mask=tgt_mask  # Causal mask
        )

        # Output layer
        feat_logits, ipa_logits = self.output_proj(decoder_output)  # (batch_size, target_len, feat_dim*5), (batch_size, target_len, vocab_size)

        return feat_logits, ipa_logits
    
    def generate(self, langs, inputs, target_langs, output_featvecs = False):
        """
        Autoregressively generates the target reflex.
        Args:
            langs: (batch_size, batch_num_rows, batch_seq_len)
            inputs: (batch_size, batch_num_rows, batch_seq_len, feat_dim)
            target_langs: (batch_size,)
        Returns:
            predictions: (batch_size, max_target_len)
        """
        batch_size = inputs.size(0)

        # Encode inputs 
        encoder_output = self._encode(langs, inputs) # (batch_size, batch_num_rows * batch_seq_len, hidden_dim)

        # Initialize with [SOS] + target language embedding
        decoder_input = self.sos_embed.expand(batch_size, 1, -1) # (batch_size, 1, hidden_dim)
        target_lang_embed = self.target_lang_embed(target_langs) # (batch_size, 1, hidden_dim)
        decoder_input = decoder_input + target_lang_embed # (batch_size, 1, hidden_dim)

        # Initialize predictions with EOS tokens
        predictions = torch.full((batch_size, self.max_target_len), 
                                self.eos_token_id, device=inputs.device)
        
        if output_featvecs:
            featvec_predictions = torch.full((batch_size, self.max_target_len, self.feat_dim), 
                                3, device=inputs.device)

        for t in range(self.max_target_len):
            # Decode step
            feat_logits, ipa_logits = self._decode_step(decoder_input, encoder_output)
            
            next_token = torch.argmax(ipa_logits[:, -1, :], dim=-1) # For model output
            next_featvec = feat_logits.reshape(batch_size, -1, self.feat_dim, 5)[:, -1, :]
            next_featvec = torch.argmax(next_featvec, dim=-1) # For decoder input

            # Store prediction
            predictions[:, t] = next_token

            if output_featvecs:
                featvec_predictions[:, t] = next_featvec

            # Stop if all sequences predicted EOS
            # A feature vector containing only 3s is hardcoded to represent EOS (see tokenizer.py)
            if (next_token == self.eos_token_id).all() or (next_featvec[:, 0] == 3).all():
                break
            
            # Project feature vector to hidden_dim and add target language embedding
            next_featvec = self.feature_proj(next_featvec.unsqueeze(1).to(torch.float32)) # (batch_size, 1, hidden_dim)
            next_featvec = next_featvec + target_lang_embed # (batch_size, 1, hidden_dim)
            
            # Update decoder input
            decoder_input = torch.cat([
                decoder_input, 
                next_featvec
            ], dim=1)

        if output_featvecs:
            return featvec_predictions
        return predictions

# Classification head for simultaneous feature vector and phoneme prediction
class PhonemeDecoder(nn.Module):
    def __init__(self, hidden_dim, feat_dim, output_dim, dropout = 0.5):
        super().__init__()
        intermediate_dim = hidden_dim * 2
        self.intermediate = nn.Linear(hidden_dim, intermediate_dim)
        self.intermediate_act = nn.ReLU()

        self.dense = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

        self.feat_head = nn.Linear(hidden_dim, feat_dim)
        self.ipa_head = nn.Linear(feat_dim, output_dim)

    def forward(self, x):
        intermediate = self.intermediate_act(self.intermediate(x))
        hidden_states = self.dropout(self.dense(intermediate))
        x = self.ln(hidden_states + x)
        feat_logits = self.feat_head(x)  # (batch_size, seq_len, feat_dim)
        ipa_logits = self.ipa_head(feat_logits)  # (batch_size, seq_len, output_dim)
        return feat_logits, ipa_logits