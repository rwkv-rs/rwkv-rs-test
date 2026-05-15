import os
import torch.nn as nn
from .ffn import RWKV_Cmix_v7
from .att import RWKV_Tmix_v7
from rwkvt.infctx_module import BlockState
from rwkvt.trace import trace
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_v7(args, layer_id)  
        self.ffn = RWKV_Cmix_v7(args, layer_id)


    # def forward(self, x, v_first):
    #     if self.layer_id == 0:
    #         x = self.ln0(x)

    #     x_attn, v_first = self.att(self.ln1(x), v_first)
    #     x = x + x_attn

    #     x = x + self.ffn(self.ln2(x))
    #     return x, v_first
    @property
    def _use_infctx(self):
        """判断是否使用无限上下文模式"""
        return os.environ.get("RWKV_TRAIN_TYPE") == 'infctx'

    def forward(self, *args, **kwargs):
        if self._use_infctx:
            return self.forward_infctx(*args, **kwargs)
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, x, v_first, attention_mask = None):
        if self.layer_id == 0:
            x = trace("layer_norm0", lambda: self.ln0(x), outputs="layer_norm0/embedded_context.safetensors")

        x_tmix = trace(
            f"cells/cell_{self.layer_id:04d}/pre_layer_norm_for_time_mix",
            lambda: self.ln1(x),
        )

        def time_mixer_forward():
            x_attn, next_v_first = self.att(x_tmix, v_first, attention_mask = attention_mask)
            value_trace = next_v_first if self.layer_id == 0 else next_v_first.contiguous().clone()
            return x_attn, next_v_first, value_trace

        x_attn, v_first, _ = trace(
            f"cells/cell_{self.layer_id:04d}/time_mixer",
            time_mixer_forward,
            outputs={
                0: f"cells/cell_{self.layer_id:04d}/time_mixer/embedded_context.safetensors",
                2: f"cells/cell_{self.layer_id:04d}/time_mixer/value_from_first_cell.safetensors",
            },
        )
        x = trace(
            f"cells/cell_{self.layer_id:04d}/embedded_context_after_time_mixer",
            lambda: x + x_attn,
            outputs=f"cells/cell_{self.layer_id:04d}/embedded_context_after_time_mixer.safetensors",
        )

        x_cmix = trace(
            f"cells/cell_{self.layer_id:04d}/pre_layer_norm_for_channel_mix",
            lambda: self.ln2(x),
        )
        x_ffn = trace(
            f"cells/cell_{self.layer_id:04d}/channel_mixer",
            lambda: self.ffn(x_cmix, attention_mask = attention_mask),
            outputs=f"cells/cell_{self.layer_id:04d}/channel_mixer/embedded_context.safetensors",
        )
        x = trace(
            f"cells/cell_{self.layer_id:04d}/embedded_context_after_channel_mixer",
            lambda: x + x_ffn,
            outputs=f"cells/cell_{self.layer_id:04d}/embedded_context_after_channel_mixer.safetensors",
        )
        return x, v_first

    def forward_infctx(self, x, v_first, last_state: BlockState, attention_mask = None):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first, att_state = self.att(self.ln1(x), v_first, last_state.time_mix_state, attention_mask = attention_mask)
        x = x + x_attn

        ffn_out ,ffn_state = self.ffn(self.ln2(x), last_state.channel_mix_state, attention_mask = attention_mask)

        x = x + ffn_out
        return x, v_first, BlockState(att_state, ffn_state)
