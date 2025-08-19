from enum import Flag
import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            #shape of key_cache: [batch_size, num_heads_kv, seq_len, head_dim]
            return self.key_cache[0].shape[-2]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        if len(self.key_cache) <= layer_idx:
            #if we never added any key and values states to the cache, create it 
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            #otherwise we concatenate the new keys with the existing one
            #each tensor has shape: [batch_size, num_key_value_heads, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

            

class GemmaConfig():
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig():
    def __init__(
        self,
        vision_config =None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_indefx =ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.text_config = text_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config =  SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id= pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size //self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size : int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))  
    
    def _norm(self, x):
        #calculate the RMS term 
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.up_proj= nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)

    def forward(self, x):
        #Equivalent to:
        # y = self.gate_proj(x), [batch_size, seq_len, hidden_size] --> [batch_size, seq_len, intermediate_size]
        # y = torch.gelu(y, approximate="tanh") -> [batch_size, seq_len, intermediate_size]
        # j = self.up_proj(x) [batch_size, seq_len, hidden_size] --> [batch_size, seq_len, intermediate_size]
        # z = y* j [batch_size, seq_len, intermediate_size]
        # z = self.down_proj(z) [batch_size, seq_len, intermediate_size] --> [batch_size, seq_len, hidden_size]
        return self.down_prj(nn.functional.gelu(self.gate_prj(x), approximate = "tanh") * self.up_proj(x))


class GemmaAttention(nn.Module):
    def __init__(self, config:GemmaConfig, layer_idx:Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "Cannot divide hidden dimensions to multiple heads "

        # Num_heads = 8
        # hidden_size = 1024
        # head_dim = 1024 //8 = 128
        # W_q = [1024, 8*128] , W_k = [1024, 1*128] , W_v = [1024, 1*128]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor]= None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
            **kwargs
            ):
        batch_size, q_len, _ = hidden_states.size() #[batch_size, seq_len, hidden_size]
        query_states = self.q_proj(hidden_states) #[batch_size, seq_len, num_heads * head_dim]
        key_states = self.k_proj(hidden_states) #[batch_size, seq_len, num_key_value_heads * head_dim]
        value_states = self.v_proj(hidden_states)  #[batch_size, seq_len, num_key_value_heads * head_dim]
        #[batch_size, seq_len, num_heads * head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        #[batch_size, seq_len, num_key_value_heads * head_dim] -> [batch_size, num_key_value_heads, seq_len, head_dim]
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

         # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #perform the calculation : attn = Q @ K.T / sqrt(head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        #Apply the softmax
        #[batch_size, Num_Heads_Q, seq_len_q, seq_len_kv]
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype= torch.float32).to(query_states.dtype)
        #Apply the dropout 
        attn_weights = nn.functional.dropout(attn_weights, p = self.attention_dropout, training = self.training)
        #[batch_size, Num_Heads_Q, seq_len_q, head_dim]
        out = torch.matmul(attn_weights, value_states)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, q_len, -1)
        #[batch_size, seq_len_q, hidden_dim]
        out = self.o_proj(out)
        return out, attn_weights

        
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)
        self.mlp = GemmaMLP(config)
        
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
    
    def forward(self, 
        hidden_states:torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states
        #[batch_size, seq_len, hidden_dim]
        hidden_states = self.input_layernorm(hidden_states)
        
        #[batch_size, seq_len, hidden_dim]
        hidden_states = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache
        )
        hidden_states = residual + hidden_states
        #[batch_size, seq_len, hidden_dim]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.mlp(hidden_states)
        return hidden_states

class GemmaModel(nn.Module):
    def __init__(self, config:GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        #[batch_size, seq_len, hidden_dim]
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype = hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            #[batch_size, seq_len, hidden_dim]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache
            )
            
        hidden_states = self.norm(hidden_states)
        return hidden_states 
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weights
    
    def forward(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache]= None,
    ):
        
        #inut_embeds: [Batch_size, seq_len, hidden_size]
        # outputs: [Batch_size, seq_len, hidden_size]
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache=kv_cache,
        )
        
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return_data = {
            "logits": logits
        }
        
        if kv_cache is not None:
            #return the updated cache
            return_data["kv_cache"] = kv_cache
        
        return return_data
    
    
    
    
    
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias = True)
    def forward(self, image_features):
        # [B, Num_patches, Embed_dim] --> [B, Num_patches, Projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id=self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self): 
        return self.language_model.tie_wieghts()

    def _merge_input_ids_with_image_features(
            self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids:torch.Tensor, attention_mask:torch.Tensor, kv_cache
    ):
        _, _, embed_dim = image_features.shape
        batch_size, seq_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5 )

        #Combine the image token, the text tokens and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, seq_length, embed_dim, dtype = inputs_embeds.dtype, device = inputs_embeds.device)
        #Shape: [batch_size, seq_len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        #shape: [batch_size, seq_len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        #shape: [batch_size, seq_len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id
        #We need to expand the masks to the embedding dimesions: [batch_size, seq_len] --> [batch_size, seq_len, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        
        #Add the text embedding 
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        #insert image embeddings.We can't use torch.where because the sequence length of scaled_image_features is not equal to the seq_length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features) 
        #Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        
        #CREATE ATTENTION MASK 
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            #Do not mask any tokens, we are in a prefill stage
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            #Since we are generating tokens, the query must be only one token
            assert q_len==1
            kv_len = kv_cache.num_items() + q_len
            #Dont need to mask anything, since each query should be able to attend to all previous tokens 
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value = 0, dtype = dtype, device = device
            )
        #[batch_size, q_len, kv_len] --> [batch_size, num_head_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)
         
        if kv_cache is not None and kv_cache.num_items()>0:
             #the position of the query is the last position
             position_ids = attention_mask.cumsum(-1)[:, -1]
             if position_ids.dim() == 1:
                 position_ids = position_ids.unsqueeze(0) # [batch_size, seq_len]
        else:
            #Create a position_ids bassed on the size of the attention_mask
            # For masked_tokens, use the number 1 as position 
            position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask==0), 1).to(device)
                 
        return final_embedding, causal_mask, position_ids 


    def forward(self,
                inputs_ids: torch.LongTensor,
                pixel_values: torch.FloatTensor,
                attention_mask: Optional[torch.Tensor],
                kv_cache: Optional[KVCache]):
        assert torch.all(attention_mask == 1), "the input can not be padded"

        #Extract input embeddings
        #[Batch, Seq_len, Embed_dim]
        inputs_embeds = self.language_model.get_input_embeddings()(inputs_ids)

        #Merge text and images
        #[B, C, H, W] --> [B, Num_patches, Embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds))
        #[B, Num_patches, Embed_dim] --> [B, Num_patches, hidden_dim]
        image_features = self.multi_modal_projector(selected_image_feature)

        #merge the embeddings of the text tokens and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache
        )
        return outputs
