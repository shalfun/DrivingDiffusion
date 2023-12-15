import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention import CrossAttention
from einops import rearrange


class ConsistencyAttention(CrossAttention):
    """
    Custom attention mechanism that extends CrossAttention with additional features
    and configurations for handling various architectural modes.
    """
    
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        arch_mode: str = "Default",
    ):
        super().__init__(
            query_dim, cross_attention_dim, heads, dim_head, dropout,
            bias, upcast_attention, upcast_softmax, added_kv_proj_dim, norm_num_groups)
        self.arch_mode = arch_mode

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError("[{}] Added KV projection dimension is not implemented.".format(self.__class__.__name__))

        key, value = self.prepare_kv(encoder_hidden_states if encoder_hidden_states is not None else hidden_states, video_length)

        if attention_mask is not None:
            attention_mask = self.process_attention_mask(attention_mask, query.shape[1], self.heads)

        hidden_states = self.compute_attention(query, key, value, attention_mask, sequence_length)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

    def process_tensor_temporal(self, tensor, video_length, ind):
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=video_length)
        tensor = torch.cat([tensor[:, [0] * video_length], tensor[:, ind]], dim=2)
        return rearrange(tensor, "b f d c -> (b f) d c")

    def process_tensor_view(self, tensor, video_length, ind_1, ind_2):
        tensor = rearrange(tensor, "(b f) d c -> b f d c", f=video_length)
        tensor = torch.cat([tensor[:, ind_1], tensor[:, ind_2]], dim=2)
        return rearrange(tensor, "b f d c -> (b f) d c")

    def prepare_kv(self, encoder_hidden_states, video_length):
        """
        Prepare key and value tensors based on the architectural mode.
        """
        # Implementation of key and value preparation based on self.arch_mode
        # This is a placeholder implementation; you should modify it according to your needs.
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if 'video' in self.arch_mode or 'autoreg' in self.arch_mode: #tricky
            channel_index_1 = torch.arange(video_length) - 1
            channel_index_1[0] = 0 # [ 0 0 1 2 3 4]
            key = self.process_tensor_temporal(key, video_length, channel_index_1)
            value = self.process_tensor_temporal(value, video_length, channel_index_1)

        elif 'img' in self.arch_mode: # single frame
            if '6view' in self.arch_mode:
                # ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT']
                channel_index_1 = torch.tensor([5,0,1,2,3,4])
                channel_index_2 = torch.tensor([1,2,3,4,5,0])
            elif '3view' in self.arch_mode:
                # ["spherical_left_forward", "onsemi_obstacle", "spherical_right_forward"]
                channel_index_1 = torch.tensor([0,0,1])
                channel_index_2 = torch.tensor([1,2,2])
            else:
                raise ValueError("[{}] How many view".format(self.__class__.__name__))

            key = self.process_tensor_view(key, video_length, channel_index_2, channel_index_1)
            value = self.process_tensor_view(value, video_length, channel_index_2, channel_index_1)

        else:
            print('unknown arch_mode:', self.arch_mode)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        
        return key, value

    def process_attention_mask(self, attention_mask, target_length, num_heads):
        """
        Process the attention mask to match the target length and head count.
        """
        if attention_mask.shape[-1] != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length - attention_mask.shape[-1]), value=0.0)
            attention_mask = attention_mask.repeat_interleave(num_heads, dim=0)
        return attention_mask

    def compute_attention(self, query, key, value, attention_mask, sequence_length):
        """
        Compute the attention mechanism.
        """
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, query.shape[-1], attention_mask)

        return hidden_states
