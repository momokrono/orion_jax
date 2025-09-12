import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx


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
        x = jax.nn.mish(self.bn1(self.conv1(x), use_running_average=run)).astype(self.compute_dtype)
        x = self.bn2(self.conv2(x), use_running_average=run)
        x = self.coord_att(x, run=run)
        x += residual
        return jax.nn.mish(x).astype(self.compute_dtype)


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

        self.bottleneck = []
        for _ in range(bottleneck_depth):
            self.bottleneck.append(ResidualBlock(
                filters[-1],
                filters[-1],
                3,
                1,
                compute_dtype=compute_dtype,
                rngs=rngs
            ))

        self.decoder = []
        self.upsample = []
        self.skip_attn = []
        for f in reversed(filters):
            self.upsample.append(UpsampleBlock(in_ch, f, 2, compute_dtype=compute_dtype, rngs=rngs))
            self.skip_attn.append(CoordAttn(f, compute_dtype=compute_dtype, rngs=rngs))
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
        for block in self.bottleneck:
            x = block(x, run=run)
        skip_connections.reverse()
        for block, upscale, skip_att, connection in zip(self.decoder, self.upsample, self.skip_attn, skip_connections):
            x = upscale(x)
            att = skip_att(connection)
            x = jnp.concatenate([x, att], axis=-1)
            x = block(x)
        output = self.output(x)
        output = input_image - output
        return output
