import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx


class FocusedLinearAttn(nnx.Module):
    def __init__(self, channels, heads: int = 4, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        assert channels % heads == 0, f"Channels {channels} must be divisible by heads {heads}"
        self.heads = heads
        self.head_dim = channels // heads
        self.compute_dtype = compute_dtype

        self.to_qkv = nnx.Conv(channels, channels * 3, kernel_size=1, rngs=rngs, dtype=compute_dtype)
        self.proj_out = nnx.Conv(channels, channels, kernel_size=1, rngs=rngs, dtype=compute_dtype)

        # Learnable spatial focus parameters
        self.gamma = nnx.Variable(jnp.ones((1, 1, 1, channels), dtype=compute_dtype))
        self.beta = nnx.Variable(jnp.zeros((1, 1, 1, channels), dtype=compute_dtype))

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        N, H, W, C = x.shape
        L = H * W  # sequence length

        identity = x

        # Project to QKV
        qkv = self.to_qkv(x)  # [N, H, W, 3*C]
        qkv = qkv.reshape(N, H, W, 3, self.heads, self.head_dim)
        qkv = qkv.transpose(3, 0, 4, 1, 2, 5)  # [3, N, heads, H, W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [N, heads, H, W, head_dim]

        # Flatten spatial dimensions: [N, heads, H*W, head_dim]
        q = q.reshape(N, self.heads, L, self.head_dim)
        k = k.reshape(N, self.heads, L, self.head_dim)
        v = v.reshape(N, self.heads, L, self.head_dim)

        # Normalize along token dimension (like softmax over sequence)
        q = jax.nn.softmax(q, axis=-1)  # normalize over feature dim (linear attention trick)
        k = jax.nn.softmax(k, axis=-2)  # normalize over token dim

        # Linear attention: (K^T @ V) then Q @ that
        context = jnp.einsum('nhlk,nhld->nhkd', k, v)  # [N, heads, head_dim, head_dim]
        out = jnp.einsum('nhkd,nhlk->nhld', context, q)  # [N, heads, L, head_dim]

        # Reshape back to spatial
        out = out.reshape(N, self.heads, H, W, self.head_dim)
        out = out.transpose(0, 2, 3, 1, 4).reshape(N, H, W, C)  # [N, H, W, C]

        out = self.proj_out(out)

        # Spatial focusing modulation
        out = out * jax.nn.sigmoid(self.gamma.value) + self.beta.value

        return identity + out  # residual connection


class CoordAttn(nnx.Module):
    def __init__(self, channels, reduction: int = 32, compute_dtype = jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.channels = channels
        self.reduction = reduction
        mip = max(8, channels // reduction)

        self.conv1 = nnx.Conv(channels, mip, kernel_size=1, strides=1, rngs=rngs, use_bias=False, dtype=compute_dtype)
        self.bn1 = nnx.BatchNorm(mip, dtype=compute_dtype, rngs=rngs, use_running_average=True)

        self.conv_h = nnx.Conv(mip, channels, kernel_size=1, strides=1, rngs=rngs, dtype=compute_dtype)
        self.conv_w = nnx.Conv(mip, channels, kernel_size=1, strides=1, rngs=rngs, dtype=compute_dtype)

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        identity = x
        n, h, w, c = x.shape
        x_h = jnp.mean(x, axis=1, keepdims=True, dtype=self.compute_dtype)
        x_w = jnp.mean(x, axis=2, keepdims=True, dtype=self.compute_dtype)
        x_w_t = x_w.transpose(0, 2, 1, 3)
        y = jnp.concatenate([x_h, x_w_t], axis=2)
        y = self.bn1(self.conv1(y), use_running_average=run)
        y = jax.nn.hard_swish(y).astype(self.compute_dtype)
        x_h_att, x_w_att = jnp.split(y, [w], axis=2)
        a_h = jax.nn.sigmoid(self.conv_h(x_h_att)).astype(self.compute_dtype)
        a_w = jax.nn.sigmoid(self.conv_w(x_w_att)).astype(self.compute_dtype)
        out = identity * a_w * a_h
        return out


class ResidualBlock(nnx.Module):
    """The basic residual block"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, compute_dtype, rngs: nnx.Rngs):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            kernel_dilation=dilation,
            use_bias=False,
            dtype=compute_dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(out_channels, use_running_average=True, dtype=compute_dtype, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            kernel_dilation=dilation,
            use_bias=False,
            dtype=compute_dtype,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(out_channels, use_running_average=True, dtype=compute_dtype, rngs=rngs)
        self.shortcut = nnx.Sequential(
            nnx.Conv(in_channels, out_channels, kernel_size=1, kernel_dilation=1, dtype=compute_dtype, rngs=rngs),
            nnx.BatchNorm(out_channels, dtype=compute_dtype, rngs=rngs)
        )
        self.coord_att = CoordAttn(out_channels, compute_dtype=compute_dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        residual = self.shortcut(x)
        x = nnx.swish(self.bn1(self.conv1(x), use_running_average=run)).astype(self.compute_dtype)
        x = self.bn2(self.conv2(x), use_running_average=run)
        x = self.coord_att(x, run=run)
        x += residual
        return nnx.swish(x).astype(self.compute_dtype)


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
    def __init__(self, embed_dim: int, num_heads: int, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.compute_dtype = compute_dtype

        # Multi-head Self-Attention
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

        # MLP
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
        x = x + x_attn

        # MLP
        x_norm = self.ln2(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        return x.astype(self.compute_dtype)


class HybridBottleneck(nnx.Module):
    def __init__(self, channels, num_heads, depth, spatial_size, compute_dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.depth = depth
        self.spatial_size = spatial_size
        self.compute_dtype = compute_dtype
        self.layers = []

        # Learnable 2D positional embeddings — shape [1, H, W, C]
        pos_embed_shape = (1, spatial_size, spatial_size, channels)
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs(), pos_embed_shape, dtype=compute_dtype) * 0.02
        )

        for i in range(depth):
            # ViT Layer (global context)
            vit_layer = ViTLayer(channels, num_heads, compute_dtype, rngs)
            # Residual Layer
            conv_block = ResidualBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                dilation=1,
                compute_dtype=compute_dtype,
                rngs=rngs
            )
            self.layers.append((vit_layer, conv_block))

    def __call__(self, x: jnp.ndarray, run: bool = True) -> jnp.ndarray:
        N, H, W, C = x.shape
        assert H == self.spatial_size and W == self.spatial_size, f"Expected {self.spatial_size}x{self.spatial_size}, got {H}x{W}"

        # Add positional embeddings — BROADCASTS to [N, H, W, C]
        x = x + self.pos_embed.value

        # Pass through hybrid layers: ViT → ResidualBlock
        for vit_layer, conv_block in self.layers:
            # --- ViT Path ---
            x_flat = x.reshape(N, H * W, C)  # [N, L, C], flattened
            x_vit = vit_layer(x_flat, run=run)
            x_vit = x_vit.reshape(N, H, W, C)  # back to spatial
            x = x + x_vit  # residual

            # --- Conv Path ---
            x_conv = conv_block(x, run=run)
            x = x + x_conv

        return x.astype(self.compute_dtype)



class Orion(nnx.Module):
    """Main Network Architecture"""
    def __init__(self,
                 in_channels: int = 3,
                 bottleneck_depth: int = 2,
                 compute_dtype: jax.typing.DTypeLike = jnp.float32,
                 rngs: nnx.Rngs = jax.random.PRNGKey(0)
                 ):
        super().__init__()
        filters = [64, 128, 256, 512]
        dilations = [1, 2, 2, 3]
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        self.encoder = []
        self.pools = []
        in_ch = in_channels
        for f, d in zip(filters, dilations):
            self.encoder.append(ResidualBlock(
                in_ch,
                f,
                kernel_size=3,
                dilation=1,
                compute_dtype=compute_dtype,
                rngs=rngs
            ))
            in_ch = f

        bottleneck_spatial_size = 256 // (2 ** 4)  # = 16
        self.bottleneck = HybridBottleneck(
            channels=filters[-1],
            num_heads=8,
            depth=bottleneck_depth,
            spatial_size=bottleneck_spatial_size,
            compute_dtype=compute_dtype,
            rngs=rngs
        )

        self.decoder = []
        self.upsample = []
        self.skip_attn = []
        for f in reversed(filters):
            self.upsample.append(UpsampleBlock(in_ch, f, 2, compute_dtype=compute_dtype, rngs=rngs))
            # self.skip_attn.append(CoordAttn(f, compute_dtype=compute_dtype, rngs=rngs))
            self.skip_attn.append(FocusedLinearAttn(f, compute_dtype=compute_dtype, rngs=rngs))
            self.decoder.append(ResidualBlock(
                f*2,
                f,
                3,
                1,
                compute_dtype,
                rngs=rngs
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
            att = skip_att(connection)
            x = jnp.concatenate([x, att], axis=-1)
            x = block(x)
        output = self.output(x)
        output = input_image - output
        return output
