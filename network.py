import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx


class SEGate(nnx.Module):
    def __init__(self, channels: int, reduction: int = 16, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.compute_dtype = compute_dtype
        hidden = max(8, channels // reduction)
        self.fc1 = nnx.Linear(channels, hidden, dtype=compute_dtype, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, channels, dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        # x: [N,H,W,C]
        n, h, w, c = x.shape
        s = jnp.mean(x, axis=(1,2), keepdims=False).astype(self.compute_dtype)  # [N,C]
        s = nnx.swish(self.fc1(s)).astype(self.compute_dtype)
        s = jax.nn.sigmoid(self.fc2(s)).astype(self.compute_dtype)  # [N,C]
        s = s[:, None, None, :]
        return x * s


class DropPath(nnx.Module):
    def __init__(self, drop_prob: float = 0.0, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.rngs = rngs

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        if (not run) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Broadcast mask over all non-batch dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(self.rngs(), keep_prob, shape).astype(x.dtype)
        return x * mask / keep_prob


class SimpleDropout(nnx.Module):
    def __init__(self, rate: float = 0.0, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.rate = float(rate)
        self.rngs = rngs
    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        if (not run) or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(self.rngs(), keep_prob, x.shape).astype(x.dtype)
        return x * mask / keep_prob


class SimpleGate(nnx.Module):
    """NAFNet's activation replacement: split channels in half and multiply."""

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return x1 * x2


class SimpleChannelAttention(nnx.Module):
    """NAFNet's SCA: a single learned per-channel scale. No sigmoid, no BatchNorm."""

    def __init__(self, channels: int, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.sca = nnx.Conv(channels, channels, kernel_size=1,
                            dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        return x * self.sca(s)


class NAFBlock(nnx.Module):
    """NAFNet residual block (activation-free, LayerNorm-based).

    Drop-in replacement for the old ResidualBlock. SimpleGate replaces nonlinear
    activations and LayerNorm replaces BatchNorm, so there is no train/eval mode
    to invert. Layer-scale residuals (beta1/beta2) are initialized to zero for
    stable training of deeper stacks.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0),
                 expansion: int = 2, dropout_rate: float = 0.0):
        super().__init__()
        self.compute_dtype = compute_dtype
        mid = in_channels * expansion

        self.norm1 = nnx.LayerNorm(in_channels, dtype=compute_dtype, rngs=rngs)
        self.c1 = nnx.Conv(in_channels, mid * 2, kernel_size=1,
                           dtype=compute_dtype, rngs=rngs)
        self.dw1 = nnx.Conv(mid * 2, mid * 2, kernel_size=3, padding='SAME',
                            feature_group_count=mid * 2,
                            dtype=compute_dtype, rngs=rngs)
        self.sg1 = SimpleGate()
        self.sca = SimpleChannelAttention(mid, compute_dtype=compute_dtype, rngs=rngs)
        self.c2 = nnx.Conv(mid, out_channels, kernel_size=1,
                           dtype=compute_dtype, rngs=rngs)
        self.beta1 = nnx.Param(jnp.zeros((1, 1, 1, out_channels), dtype=compute_dtype))

        ff = out_channels * expansion
        self.norm2 = nnx.LayerNorm(out_channels, dtype=compute_dtype, rngs=rngs)
        self.f1 = nnx.Conv(out_channels, ff * 2, kernel_size=1,
                           dtype=compute_dtype, rngs=rngs)
        self.dw2 = nnx.Conv(ff * 2, ff * 2, kernel_size=3, padding='SAME',
                            feature_group_count=ff * 2,
                            dtype=compute_dtype, rngs=rngs)
        self.sg2 = SimpleGate()
        self.f2 = nnx.Conv(ff, out_channels, kernel_size=1,
                           dtype=compute_dtype, rngs=rngs)
        self.beta2 = nnx.Param(jnp.zeros((1, 1, 1, out_channels), dtype=compute_dtype))

        self.dropout = SimpleDropout(dropout_rate, rngs=rngs)
        self.shortcut = (
            nnx.Conv(in_channels, out_channels, kernel_size=1,
                     dtype=compute_dtype, rngs=rngs)
            if in_channels != out_channels else None
        )

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        identity = x if self.shortcut is None else self.shortcut(x)

        h = self.norm1(x)
        h = self.c1(h)
        h = self.dw1(h)
        h = self.sg1(h)
        h = self.sca(h)
        h = self.c2(h)
        h = self.dropout(h, run=run)
        x = identity + self.beta1 * h

        h = self.norm2(x)
        h = self.f1(h)
        h = self.dw2(h)
        h = self.sg2(h)
        h = self.f2(h)
        h = self.dropout(h, run=run)
        x = x + self.beta2 * h
        return x.astype(self.compute_dtype)


class PixelShuffle(nnx.Module):
    """PixelShuffle upsampling module"""
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r = self.upscale_factor
        N, H, W, C = x.shape

        assert C % (r ** 2) == 0, f"Channels {C} must be divisible by {r ** 2}"
        x = x.reshape(N, H, W, r, r, C // (r ** 2))
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(N, H * r, W * r, -1)
        return x


class UpsampleBlock(nnx.Module):
    """Upsampling block using the PixelShuffle upsampling module"""
    def __init__(self, in_channels, out_channels, upscale_factor, compute_dtype, rngs: nnx.Rngs):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.conv = nnx.Conv(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            kernel_dilation=1,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.pixel_shuffle = PixelShuffle(upscale_factor)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = nnx.swish(x).astype(self.compute_dtype)
        return x


class ViTLayer(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0), attn_dropout_rate: float = 0.0, mlp_dropout_rate: float = 0.0):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.attn_dropout = SimpleDropout(attn_dropout_rate, rngs=rngs)
        self.mlp_dropout = SimpleDropout(mlp_dropout_rate, rngs=rngs)

        self.mhsa = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
            decode=False,
        )
        self.ln1 = nnx.LayerNorm(embed_dim, dtype=compute_dtype, rngs=rngs)

        mlp_hidden = embed_dim * 4
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_hidden, dtype=compute_dtype, rngs=rngs),
            lambda x: nnx.swish(x).astype(compute_dtype),
            nnx.Linear(mlp_hidden, embed_dim, dtype=compute_dtype, rngs=rngs)
        )
        self.ln2 = nnx.LayerNorm(embed_dim, dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, run: bool = False) -> jnp.ndarray:
        # x: [N, L, C]
        # Self-Attention
        x_norm = self.ln1(x)
        x_attn = self.mhsa(x_norm, x_norm, x_norm)  # Q=K=V
        x_attn = self.attn_dropout(x_attn, run=run)
        x = x + x_attn

        # MLP
        x_norm = self.ln2(x)
        x_mlp = self.mlp(x_norm)
        x_mlp = self.mlp_dropout(x_mlp, run=run)
        x = x + x_mlp

        return x.astype(self.compute_dtype)


class HybridLayer(nnx.Module):
    """A single hybrid ViT-conv layer used inside HybridBottleneck."""

    def __init__(self, channels, num_heads, drop_rate, compute_dtype, rngs,
                 attn_dropout_rate, mlp_dropout_rate, conv_dropout_rate, naf_expansion):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.vit = ViTLayer(channels, num_heads, compute_dtype, rngs,
                            attn_dropout_rate=attn_dropout_rate, mlp_dropout_rate=mlp_dropout_rate)
        self.conv = NAFBlock(channels, channels, compute_dtype=compute_dtype, rngs=rngs,
                             expansion=naf_expansion, dropout_rate=conv_dropout_rate)
        self.scale = nnx.Param(jnp.ones(channels, dtype=compute_dtype) * 0.1)
        self.drop_path = DropPath(drop_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        N, H, W, C = x.shape
        x_flat = x.reshape(N, H * W, C)
        x_vit = self.vit(x_flat, run=run).reshape(N, H, W, C)
        x = x + self.drop_path(x_vit * self.scale, run=run)
        x = x + self.conv(x, run=run)
        return x


class HybridBottleneck(nnx.Module):
    def __init__(self, channels, num_heads, depth, spatial_size, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0), attn_dropout_rate: float = 0.0, mlp_dropout_rate: float = 0.0, stochastic_depth_rate: float = 0.0, conv_dropout_rate: float = 0.0, naf_expansion: int = 2):
        super().__init__()
        self.depth = depth
        self.spatial_size = spatial_size
        self.compute_dtype = compute_dtype
        self.layers = nnx.List()

        # Learnable 2D positional embeddings — shape [1, H, W, C]
        pos_embed_shape = (1, spatial_size, spatial_size, channels)
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs(), pos_embed_shape, dtype=compute_dtype) * 0.02
        )

        for i in range(depth):
            rate = float(stochastic_depth_rate) * float(i + 1) / float(max(1, depth))
            self.layers.append(HybridLayer(
                channels=channels, num_heads=num_heads, drop_rate=rate,
                compute_dtype=compute_dtype, rngs=rngs,
                attn_dropout_rate=attn_dropout_rate, mlp_dropout_rate=mlp_dropout_rate,
                conv_dropout_rate=conv_dropout_rate, naf_expansion=naf_expansion,
            ))

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        N, H, W, C = x.shape
        assert H == self.spatial_size and W == self.spatial_size, f"Expected {self.spatial_size}x{self.spatial_size}, got {H}x{W}"

        x_in = x
        x = x + self.pos_embed.value

        for layer in self.layers:
            x = layer(x, run=run)

        x = x + x_in

        return x.astype(self.compute_dtype)


class Orion(nnx.Module):
    """Main Network Architecture"""
    def __init__(self,
                 in_channels: int = 3,
                 bottleneck_depth: int = 2,
                 compute_dtype: jax.typing.DTypeLike = jnp.float32,
                 rngs: nnx.Rngs = jax.random.PRNGKey(0),
                 # Regularization
                 vit_mlp_dropout_rate: float = 0.1,
                 attn_dropout_rate: float = 0.0,
                 stochastic_depth_rate: float = 0.05,
                 conv_dropout_rate: float = 0.05,
                 naf_expansion: int = 2,
                 # Input spatial size (needed for bottleneck positional embeddings)
                 input_size: int = 256,
                 ):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        self.encoder = nnx.List()
        in_ch = in_channels
        for f in filters:
            self.encoder.append(NAFBlock(
                in_ch,
                f,
                compute_dtype=compute_dtype,
                rngs=rngs,
                expansion=naf_expansion,
                dropout_rate=conv_dropout_rate,
            ))
            in_ch = f

        # Calculate bottleneck spatial size based on number of pooling layers
        num_pooling_layers = len(filters)
        bottleneck_spatial_size = input_size // (2 ** num_pooling_layers)
        self.bottleneck = HybridBottleneck(
            channels=filters[-1],
            num_heads=8,
            depth=bottleneck_depth,
            spatial_size=bottleneck_spatial_size,
            compute_dtype=compute_dtype,
            rngs=rngs,
            attn_dropout_rate=attn_dropout_rate,
            mlp_dropout_rate=vit_mlp_dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate,
            conv_dropout_rate=conv_dropout_rate,
            naf_expansion=naf_expansion,
        )

        self.decoder = nnx.List()
        self.upsample = nnx.List()
        self.skip_attn = nnx.List()
        for f in reversed(filters):
            self.upsample.append(UpsampleBlock(in_ch, f, 2, compute_dtype=compute_dtype, rngs=rngs))
            self.skip_attn.append(SEGate(f, compute_dtype=compute_dtype, rngs=rngs))
            self.decoder.append(NAFBlock(
                f * 2,
                f,
                compute_dtype=compute_dtype,
                rngs=rngs,
                expansion=naf_expansion,
                dropout_rate=conv_dropout_rate,
            ))
            in_ch = f

        self.output = nnx.Conv(in_ch, 3, kernel_size=1, kernel_dilation=1, dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        input_image = x
        skip_connections = []
        for block in self.encoder:
            x = block(x, run=run)
            skip_connections.append(x)
            x = self.avg_pool(x)
        x = self.bottleneck(x, run=run)
        skip_connections.reverse()
        for block, upscale, skip_att, connection in zip(self.decoder, self.upsample, self.skip_attn, skip_connections):
            x = upscale(x)
            att = skip_att(connection, run=run)
            x = jnp.concatenate([x, att], axis=-1)
            x = block(x, run=run)
        output = self.output(x)
        output = input_image - output
        return output
