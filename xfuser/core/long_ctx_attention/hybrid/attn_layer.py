import torch
from torch import Tensor

import torch.distributed
from yunchang import LongContextAttention
from yunchang.ring.utils import RingComm
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")

from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import HAS_SPARSE_SAGE_ATTENTION
import torch.distributed as dist
try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
    elif major>=9:
        from sageattention_sm90 import sageattn
except:
    try:
        from sageattention import sageattn
    except:
        sageattn = None


from xfuser.logger import init_logger

    

logger = init_logger(__name__)


class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
            attn_type: AttnType = AttnType.FA, the attention type supported inside long context attention, including "TORCH", "FA", "FA3", "SAGE_FP16", "SAGE_FP8"
            attn_processor: nn.Module = None, the attention processor can be passed in to replace the attention processor if attn_type is do not support it.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            use_sync=use_sync,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )

        if HAS_SPARSE_SAGE_ATTENTION:
            from spas_sage_attn.autotune import SparseAttentionMeansim
            if isinstance(attn_processor, SparseAttentionMeansim) and torch.distributed.get_world_size(self.ring_pg) > 1:
                raise RuntimeError("Sparse Sage attention does not support ring degree > 1.")

        self.attn_processor = attn_processor
        from xfuser.core.long_ctx_attention.ring import xdit_ring_flash_attn_func, xdit_ring_sage_attn_func
        from xfuser.core.long_ctx_attention.hybrid import xdit_sage_attn_func
        comm = RingComm(self.ring_pg)
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        if sageattn is not None and comm.world_size==1 and sm>86:
            self.ring_attn_fn = xdit_sage_attn_func
        else:
            if sm>86 and sageattn is not None:
                self.ring_attn_fn = xdit_ring_sage_attn_func
            else:
                self.ring_attn_fn = xdit_ring_flash_attn_func

    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """
        is_joint = False
        if (joint_tensor_query is not None and 
            joint_tensor_key is not None and 
            joint_tensor_value is not None):
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
                )
            elif joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
                is_joint = True
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
                is_joint = True
        elif (joint_tensor_query is None and 
            joint_tensor_key is None and 
            joint_tensor_value is None):
            pass
        else:
            raise ValueError(
                f"joint_tensor_query, joint_tensor_key, and joint_tensor_value should be None or not None simultaneously."
            )

        if is_joint:
            ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
            ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            )
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_type=self.attn_type,
            attn_processor=self.attn_processor,
            attn_layer=attn if self.use_kv_cache else None,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output


class xFuserLongContextAttentionOverLap(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
            attn_type: AttnType = AttnType.FA, the attention type supported inside long context attention, including "TORCH", "FA", "FA3", "SAGE_FP16", "SAGE_FP8"
            attn_processor: nn.Module = None, the attention processor can be passed in to replace the attention processor if attn_type is do not support it.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )

        if HAS_SPARSE_SAGE_ATTENTION:
            from spas_sage_attn.autotune import SparseAttentionMeansim
            if isinstance(attn_processor, SparseAttentionMeansim) and torch.distributed.get_world_size(self.ring_pg) > 1:
                raise RuntimeError("Sparse Sage attention does not support ring degree > 1.")
        self.sp_world_size = dist.get_world_size(self.ulysses_pg)

        self.attn_processor = attn_processor
        from xfuser.core.long_ctx_attention.ring import xdit_ring_flash_attn_func
        from xfuser.core.long_ctx_attention.hybrid import xdit_sage_attn_func
        comm = RingComm(self.ring_pg)
        if sageattn is not None and comm.world_size==1:
            self.ring_attn_fn = xdit_sage_attn_func
        else:
            self.ring_attn_fn = xdit_ring_flash_attn_func

    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """
        is_joint = False
        if (joint_tensor_query is not None and 
            joint_tensor_key is not None and 
            joint_tensor_value is not None):
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
                )
            elif joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
                is_joint = True
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
                is_joint = True
        elif (joint_tensor_query is None and 
            joint_tensor_key is None and 
            joint_tensor_value is None):
            pass
        else:
            raise ValueError(
                f"joint_tensor_query, joint_tensor_key, and joint_tensor_value should be None or not None simultaneously."
            )

        if is_joint:
            ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
            ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            )
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            B, N_sub, H, D = query.shape
            SP = self.sp_world_size
            num_stages = H // SP
            head_chunk = H // SP // num_stages
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv_split = qkv.view(3, B, N_sub, SP, num_stages, head_chunk, D)
            #(num_stages, SP, 3, B, N_sub, head_chunk, D)
            qkv_split = qkv_split.permute(4, 3, 0, 1, 2, 5, 6).contiguous()
            def _all_to_all_head_chunk(stage):
                input_ = qkv_split[stage]
                output_ = torch.empty_like(input_)
                work = dist.all_to_all_single(output_, input_, async_op=True)
                return work, output_
            
            def _calculate(output_: torch.Tensor):
                #(SP , 3, B, N_sub, head_chunk, D)
                q, k, v = output_.permute(1, 2, 0, 3, 4, 5).reshape(3, B, SP * N_sub, head_chunk, D).contiguous().unbind(0)
                out = self.ring_attn_fn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    group=self.ring_pg,
                    attn_type=self.attn_type,
                    attn_processor=self.attn_processor,
                    attn_layer=attn if self.use_kv_cache else None,
                    joint_tensor_key=joint_tensor_key,
                    joint_tensor_value=joint_tensor_value,
                    joint_strategy=joint_strategy,
                )
                if type(out) == tuple:
                    context_layer, _, _ = out
                else:
                    context_layer = out
                return context_layer
            def _all_to_all_res_list(res_list):  # -> List[B, S, heads_chunk, D]
                stages = len(res_list)
                res_list = [x.permute(1, 0, 2, 3) for x in res_list]  # [S, B, heads_chunk, D]
                intpu_ = torch.stack(res_list, dim=2)  # [S, B, stages, heads_chunk, D]
                intpu_ = intpu_.view(SP, N_sub, B, stages, head_chunk, D)
                output_ = torch.empty_like(intpu_)
                work = dist.all_to_all_single(output_, intpu_, async_op=True)
                # [B, N_sub, SP, stages, heads_chunk, D]
                return work, output_.permute(2, 1, 0, 3, 4, 5)
            res = []
            dist.barrier()
            last_work, last_data = _all_to_all_head_chunk(0)
            for i in range(1, num_stages):
                work, data = _all_to_all_head_chunk(i)
                last_work.wait()
                res.append(_calculate(last_data))  # [B, S, heads_chunk, D]
                last_work = work
                last_data = data
            last_work.wait()
            # [SP, N_sub, B, stages-1, heads_chunk, D]
            if num_stages > 1:
                work, output_0 = _all_to_all_res_list(res)
            work, output_1 = _all_to_all_res_list([_calculate(last_data)])
            work.wait()
            return torch.cat([output_0, output_1], dim=3).view(B, N_sub, H, D) if res else output_1.reshape(B, N_sub, H, D)

        else:
            B, N_sub, H, D = query.shape
            SP = self.sp_world_size
            num_stages = H // SP
            head_chunk = H // SP // num_stages
            # (bs, seq_len, head_cnt/N, head_size)
            qkv_splits = [qkv.view(B, N_sub, SP, num_stages, head_chunk, D) for qkv in [query, key, value]]
            #(num_stages, SP, B, N_sub, head_chunk, D)
            qkv_splits = [qkv_split.permute(3, 2, 0, 1, 4, 5).contiguous() for qkv_split in qkv_splits]
            def _all_to_all_head_chunk(stage):
                works = []
                outputs_ = []
                for qkv_split in qkv_splits:
                    input_ = qkv_split[stage]
                    output_ = torch.empty_like(input_)
                    work = dist.all_to_all_single(output_, input_, async_op=True)
                    works.append(work)
                    outputs_.append(output_)
                return works, outputs_
            
            def _calculate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
                #(SP, B, N_sub, head_chunk, D)
                q, k, v = [_q.permute(1, 0, 2, 3, 4).reshape(B, SP * N_sub, head_chunk, D).contiguous() for _q in [q, k, v]]
                out = self.ring_attn_fn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    group=self.ring_pg,
                    attn_type=self.attn_type,
                    attn_processor=self.attn_processor,
                    attn_layer=attn if self.use_kv_cache else None,
                    joint_tensor_key=joint_tensor_key,
                    joint_tensor_value=joint_tensor_value,
                    joint_strategy=joint_strategy,
                )
                if type(out) == tuple:
                    context_layer, _, _ = out
                else:
                    context_layer = out
                return context_layer
            def _all_to_all_res_list(res_list):  # -> List[B, S, heads_chunk, D]
                stages = len(res_list)
                res_list = [x.permute(1, 0, 2, 3) for x in res_list]  # [S, B, heads_chunk, D]
                intpu_ = torch.stack(res_list, dim=2)  # [S, B, stages, heads_chunk, D]
                intpu_ = intpu_.view(SP, N_sub, B, stages, head_chunk, D)
                output_ = torch.empty_like(intpu_)
                work = dist.all_to_all_single(output_, intpu_, async_op=True)
                # [B, N_sub, SP, stages, heads_chunk, D]
                return work, output_.permute(2, 1, 0, 3, 4, 5)
            res = []
            dist.barrier()
            last_works, last_datas = _all_to_all_head_chunk(0)
            for i in range(1, num_stages):
                works, datas = _all_to_all_head_chunk(i)
                [last_work.wait() for last_work in last_works]
                res.append(_calculate(*last_datas))  # [B, S, heads_chunk, D]
                last_works = works
                last_datas = datas
            [last_work.wait() for last_work in last_works]
            # [SP, N_sub, B, stages-1, heads_chunk, D]
            if num_stages > 1:
                work, output_0 = _all_to_all_res_list(res)
            work, output_1 = _all_to_all_res_list([_calculate(*last_datas)])
            work.wait()
            return torch.cat([output_0, output_1], dim=3).view(B, N_sub, H, D) if res else output_1.reshape(B, N_sub, H, D)