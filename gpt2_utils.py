import jax
import jax.numpy as jnp
import numpy as np
import typing as tp
import functools
from flax import linen as nn  # Linen API
from ml_collections import ConfigDict

from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer
from flax.core import freeze, unfreeze

def load_params(config, state):
    hf_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2", dtype=jnp.float32)
    hf_params = unfreeze(hf_model.params)

    params = state.params
    params['token_embed']['embedding'] = hf_params['transformer']['wte']['embedding']
    # Positional embedding
    params['pos_emb'] = hf_params['transformer']['wpe']['embedding']

    # Loop through layers
    for i in range(12):
        hf_block = hf_params['transformer']['h'][str(i)]
        block = params[f'block_{i}']
        attn_key = "CheckpointAttentionBlock_0" if "CheckpointAttentionBlock_0" in block else "AttentionBlock_0"
        mlp_key = "CheckpointMLPBlock_0" if "CheckpointMLPBlock_0" in block else "MLPBlock_0"

        # LayerNorms
        block[attn_key]['LayerNorm_0']['scale'] = hf_block['ln_1']['scale']
        block[attn_key]['LayerNorm_0']['bias'] = hf_block['ln_1']['bias']
        block[mlp_key]['LayerNorm_0']['scale'] = hf_block['ln_2']['scale']
        block[mlp_key]['LayerNorm_0']['bias'] = hf_block['ln_2']['bias']

        # --- TRY 1
        # # Attention QKV and output projection
        # qkv_kernel = hf_block['attn']['c_attn']['kernel']  # shape [768, 3*768]
        # qkv_bias = hf_block['attn']['c_attn']['bias']
        # qkv_kernel = np.reshape(qkv_kernel, (config.model.hidden_size, config.model.num_heads, 3 * config.model.head_dim))
        # q_kernel, k_kernel, v_kernel = np.split(qkv_kernel, 3, axis=-1)

        # qkv_bias = np.reshape(qkv_bias, (config.model.num_heads, 3 * config.model.head_dim))
        # q_bias, k_bias, v_bias = np.split(qkv_bias, 3, axis=-1)

        # block[attn_key]['DenseGeneral_0']['kernel'] = np.concatenate([q_kernel, k_kernel, v_kernel], axis=-1)
        # block[attn_key]['DenseGeneral_0']['bias'] = np.concatenate([q_bias, k_bias, v_bias], axis=-1)

        # block[attn_key]['Dense_0']['kernel'] = hf_block['attn']['c_proj']['kernel']
        # block[attn_key]['Dense_0']['bias'] = hf_block['attn']['c_proj']['bias']

        # # MLP
        # block[mlp_key]['Dense_0']['kernel'] = hf_block['mlp']['c_fc']['kernel']
        # block[mlp_key]['Dense_0']['bias'] = hf_block['mlp']['c_fc']['bias']
        # block[mlp_key]['Dense_1']['kernel'] = hf_block['mlp']['c_proj']['kernel']
        # block[mlp_key]['Dense_1']['bias'] = hf_block['mlp']['c_proj']['bias']

        # --- TRY 2

        # # Attention: QKV projection (c_attn)
        # block[attn_key]['DenseGeneral_0']['kernel'] = hf_block['attn']['c_attn']['kernel'].T
        # block[attn_key]['DenseGeneral_0']['bias'] = hf_block['attn']['c_attn']['bias']

        # # Attention output (c_proj)
        # block[attn_key]['Dense_0']['kernel'] = hf_block['attn']['c_proj']['kernel'].T
        # block[attn_key]['Dense_0']['bias'] = hf_block['attn']['c_proj']['bias']

        # # MLP
        # block[mlp_key]['Dense_0']['kernel'] = hf_block['mlp']['c_fc']['kernel'].T
        # block[mlp_key]['Dense_0']['bias'] = hf_block['mlp']['c_fc']['bias']
        # block[mlp_key]['Dense_1']['kernel'] = hf_block['mlp']['c_proj']['kernel'].T
        # block[mlp_key]['Dense_1']['bias'] = hf_block['mlp']['c_proj']['bias']

        # --- Try 3

        # # --- Attention QKV ---
        qkv_kernel = hf_block['attn']['c_attn']['kernel'].T  # (768, 2304)
        qkv_kernel = qkv_kernel.reshape(768, 3, config.model.num_heads, config.model.head_dim)  # (768, 3, 12, 64)
        qkv_kernel = np.transpose(qkv_kernel, (0, 2, 1, 3))  # (768, 12, 3, 64)
        qkv_kernel = qkv_kernel.reshape(768, config.model.num_heads, 3 * config.model.head_dim)  # (768, 12, 192)

        qkv_bias = hf_block['attn']['c_attn']['bias'].reshape(3, config.model.num_heads, config.model.head_dim)  # (3, 12, 64)
        qkv_bias = np.transpose(qkv_bias, (1, 0, 2)).reshape(config.model.num_heads, 3 * config.model.head_dim)  # (12, 192)

        block[attn_key]['DenseGeneral_0']['kernel'] = qkv_kernel
        block[attn_key]['DenseGeneral_0']['bias'] = qkv_bias

        # --- Attention output projection ---
        block[attn_key]['Dense_0']['kernel'] = hf_block['attn']['c_proj']['kernel'].T
        block[attn_key]['Dense_0']['bias'] = hf_block['attn']['c_proj']['bias']

        # --- MLP ---
        block[mlp_key]['Dense_0']['kernel'] = hf_block['mlp']['c_fc']['kernel'].T
        block[mlp_key]['Dense_0']['bias'] = hf_block['mlp']['c_fc']['bias']
        block[mlp_key]['Dense_1']['kernel'] = hf_block['mlp']['c_proj']['kernel'].T
        block[mlp_key]['Dense_1']['bias'] = hf_block['mlp']['c_proj']['bias']

    # Final layer norm
    params['LayerNorm_0']['scale'] = hf_params['transformer']['ln_f']['scale']
    params['LayerNorm_0']['bias'] = hf_params['transformer']['ln_f']['bias']
    state = state.replace(params=params)

    assert np.allclose(
        state.params['token_embed']['embedding'],
        hf_params['transformer']['wte']['embedding']
    )

    del hf_model
    del hf_params

    return state


class MLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        x = nn.LayerNorm(dtype=self.config.model.dtype)(x)
        x = nn.Dense(self.config.model.mlp_expansion * features, dtype=self.config.model.dtype)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(features, dtype=self.config.model.dtype)(x)
        x = nn.Dropout(rate=self.config.model.dropout_rate)(x, deterministic=not self.train)
        return x


class AttentionBlock(nn.Module):
    config: ConfigDict
    mask: tp.Optional[jax.Array]
    train: bool

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        x = nn.LayerNorm(dtype=self.config.model.dtype)(x)
        qkv = nn.DenseGeneral(
            features=(self.config.model.num_heads, self.config.model.head_dim * 3),
            axis=-1, dtype=self.config.model.dtype
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        scale = q.shape[-1] ** -0.5
        q = q.astype(self.config.model.softmax_dtype) * scale
        k = k.astype(self.config.model.softmax_dtype)

        q = q.transpose(0, 2, 1, 3)  # [B T H D] to [B H T D]
        k = k.transpose(0, 2, 1, 3)  # [B T H D] to [B H T D]
        v = v.transpose(0, 2, 1, 3)  # [B T H D] to [B H T D]

        attn = q @ k.swapaxes(-2, -1)  # [B H T D] @ [B H D T] -> [B H T T]

        if self.mask is not None:
            attn = jnp.where(self.mask, attn, jnp.finfo(self.config.model.softmax_dtype).min)

        attn = nn.softmax(attn, axis=-1).astype(self.config.model.dtype)
        attn = nn.Dropout(rate=self.config.model.dropout_rate)(attn, deterministic=not self.train)
        y = attn @ v  # [B H T T] @ [B H T D] -> [B H T D]
        y = y.transpose(0, 2, 1, 3)  # [B H T D] -> [B T H D]
        y = y.reshape(x.shape)  # [B T H D] -> [B T C(H*D)]
        y = nn.Dense(features, dtype=self.config.model.dtype)(y)
        y = nn.Dropout(rate=self.config.model.dropout_rate)(y, deterministic=not self.train)
        return y


class TransformerBlock(nn.Module):
    config: ConfigDict
    mask: tp.Optional[jax.Array]
    train: bool

    @nn.compact
    def __call__(self, x):
        mlp = MLPBlock
        if "MLP" in self.config.model.remat:
            mlp = nn.remat(mlp, prevent_cse=False)
        attn = AttentionBlock
        if "Attn" in self.config.model.remat:
            attn = nn.remat(attn, prevent_cse=False)

        x = x + attn(config=self.config, mask=self.mask, train=self.train)(x)
        x = x + mlp(config=self.config, train=self.train)(x)
        return x


class Transformer(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        if mask is None and self.config.model.causal_mask:
            mask = nn.make_causal_mask(x, dtype=jnp.bool_)

        embed = nn.Embed(self.config.model.vocab_size, self.config.model.hidden_size, dtype=self.config.model.dtype, name='token_embed')
        x = embed(x)
        pos_emb = self.param("pos_emb", nn.initializers.normal(0.02),
                             (self.config.model.max_seq_len, self.config.model.hidden_size)).astype(self.config.model.dtype)

        x += pos_emb[None, :x.shape[1]]

        block_fn = functools.partial(TransformerBlock, config=self.config, mask=mask, train=train)

        if self.config.model.scan_layers:
            block = block_fn(name="block")
            x, _ = nn.scan(
                lambda module, carry, _: (module(carry), None),
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                length=self.config.model.num_layers
            )(block, x, ())
        else:
            for i in range(self.config.model.num_layers):
                x = block_fn(name=f"block_{i}")(x)

        x = nn.LayerNorm(dtype=self.config.model.dtype)(x)

        # weight tying
        logits = x @ embed.embedding.T
        return logits.astype(jnp.float32)

def token_predictions(state, sample_batch, mode='col'):

    from transformers import GPT2Tokenizer

    # token_ids = sample_batch[0]
    token_ids_in = sample_batch[0][0]
    sample_out = state.apply_fn({'params': state.params,}, sample_batch[0][0][np.newaxis, :], train=False)

    token_ids_out = jnp.argmax(nn.softmax(sample_out), axis=-1)  # shape: [batch_size, seq_len]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    decoded_in = np.array(tokenizer.batch_decode(token_ids_in, skip_special_tokens=True)).flatten()
    decoded_out = np.array(tokenizer.batch_decode(token_ids_out, skip_special_tokens=True))
    # print(decoded_in)
    # print(decoded_out)
    print("-------------------------------------------------------------------------------------------------------")
    if mode == 'col':
        print("INPUTS")
        print("".join(decoded_in.tolist()))
        # for text in decoded_in:
        #     print(text)
        print("OUTPUTS")
        # for text in decoded_out:
        #     print(text)
        print(decoded_out)
        # print(",".join(decoded_out.tolist()))

    else:
        from itertools import zip_longest
        for in_str, out_str in zip(decoded_in, decoded_out):
            in_words = in_str.split()
            out_words = out_str.split()
            for in_word, out_word in zip_longest(in_words, out_words, fillvalue=""):
                print(f"{in_word:<15} | {out_word}")
            print("-" * 40)


    print("-------------------------------------------------------------------------------------------------------")
