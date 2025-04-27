import io
import torch
import torch.nn as nn
import torch.onnx
import onnx
import pytest

# Minimal MultiheadAttention (klassisch)
class SimpleMHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        return self.mha(x, x, x)[0]

# Minimal GQA-Modell
class SimpleGQA(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim * num_kv_heads // num_query_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim * num_kv_heads // num_query_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, seq, embed_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # reshape to [B, QH/KVH, T, D]
        B, T, _ = x.shape
        D = self.embed_dim // self.num_query_heads
        q = q.view(B, T, self.num_query_heads, D).transpose(1,2)
        kv_D = self.embed_dim // self.num_query_heads
        k = k.view(B, T, self.num_kv_heads, kv_D).transpose(1,2)
        v = v.view(B, T, self.num_kv_heads, kv_D).transpose(1,2)
        # Use scaled_dot_product_attention with enable_gqa=True
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        out = out.transpose(1,2).reshape(B, T, self.embed_dim)
        return self.out_proj(out)

@pytest.mark.parametrize("batch,seq,embed_dim,num_heads", [(2,8,32,4)])
def test_mha_export(batch, seq, embed_dim, num_heads):
    torch.manual_seed(0)
    model = SimpleMHA(embed_dim, num_heads)
    x = torch.randn(batch, seq, embed_dim)
    f = io.BytesIO()
    torch.onnx.export(
        model,
        x,
        f,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
    )
    onnx_model = onnx.load_model_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model)

@pytest.mark.parametrize("batch,seq,embed_dim,num_query_heads,num_kv_heads", [(2,8,32,8,4)])
def test_gqa_export(batch, seq, embed_dim, num_query_heads, num_kv_heads):
    torch.manual_seed(0)
    model = SimpleGQA(embed_dim, num_query_heads, num_kv_heads)
    x = torch.randn(batch, seq, embed_dim)
    f = io.BytesIO()
    torch.onnx.export(
        model,
        x,
        f,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
    )
    onnx_model = onnx.load_model_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model)

# Optionale Smoke-Tests für onnxruntime/CTranslate2 könnten ergänzt werden
