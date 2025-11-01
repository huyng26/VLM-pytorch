import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor):
        _, _, height, width = (
            pixel_values.shape
        )  # [B, C, H, W] --> [B, Embed_dim, Num_Patches_H,Num_patches_W]
        # Num_Patches_H = H//patch_size and Num_Patches_W = W//patch_size
        # Convolve the kernel over the image, with no overlapping patches
        patch_embeds = self.patch_embedding(pixel_values)
        # Flatten the patches, [B, Embed_dim, Num_patches_H, Num_patches_W] -- > [B, Embed_dim, Num_patches]
        embeddings = patch_embeds.flatten(2)
        # [B, Embed_dim, Num_patches] --> [B, Num_patches, Embed_dim]
        embeddings = embeddings.transpose(1, 2)
        # Add positions embeddings to each patch. Each position encoding is a vector of size embed_dim

        embeddings = embeddings + self.position_embeddings(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_prj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_prj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_prj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor):
        # print(hidden_states)
        # [B, Num_Patches, Embed_dim] --> [B, Num_heads, Num_patches, head_dim]
        q = self.q_proj(hidden_states)
        k = self.k_prj(hidden_states)
        v = self.v_prj(hidden_states)

        batch_size, num_patches, _ = q.shape
        # [B, Num_heads, Num_patches, head_dim]
        q = q.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, Num_heads, Num_patches, Num_patches]
        # print(q.size())
        attn = torch.matmul(q, k.transpose(2, 3))
        attn = attn * self.scale
        if attn.size() != (batch_size, self.num_heads, num_patches, num_patches):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, num_patches, num_patches)}, but is"
                f"{attn.size()}"
            )

        # Apply row-wise softmax
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        # Apply dropout during training
        attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)  # [B, Num_heads, Num_patches, head_dim]
        out = out.transpose(1, 2).contiguous().reshape(batch_size, num_patches, -1)
        out = self.out_prj(out)
        return out, attn


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        return self.fc2(hidden_states)


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        # [B, Num_patches, Embed_dim]
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor):
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        print(pixel_values.shape)
        # [B, C, H, W] --> [B, Num_Patches, Embed_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        # [Batch, C, H, W] --> [B, Num_Patches, Embed_dim]
        return self.vision_model(pixel_values)


if __name__ == "__main__":
    config = SiglipVisionConfig()
    embedding = SiglipVisionEmbedding(config)
    x = torch.rand(24, 3, 224, 224)
    model = SiglipVisionModel(config)
    print(model(x).shape)
