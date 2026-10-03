"""完整模型的多流 CUDA Graph；精确 shape 缓存与有界编译资源。"""

from collections import Counter, OrderedDict
from time import perf_counter

import torch
from torch import nn


INPUT_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


class CachedRotary(nn.Module):
    """固定位置范围的 RoPE；沿用 HF 计算缓存，避免重复三角函数与编译舍入漂移。"""

    def __init__(self, original, positions, dtype):
        super().__init__()
        self.original = original
        with torch.inference_mode():
            cos, sin = original(torch.empty(1, device=positions.device, dtype=dtype), positions)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, position_ids):
        return self.cos[:, :x.shape[1]].to(x.dtype), self.sin[:, :x.shape[1]].to(x.dtype)


class PreparedModel(nn.Module):
    """复用原有每一层，只把 CPU mask 构建移到设备，便于捕获完整前向。"""

    def __init__(self, model, max_len):
        super().__init__()
        self.model = model
        positions = torch.arange(max_len, device=next(model.parameters()).device)
        self.register_buffer("positions", positions[None], persistent=False)
        self.register_buffer("local_allowed", (positions[:, None] - positions[None, :]).abs()
                             <= model.encoder.config.local_attention // 2, persistent=False)
        for layer in model.encoder.layers:
            layer.attn.rotary_emb = CachedRotary(layer.attn.rotary_emb, self.positions, model.encoder.dtype)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        encoder = self.model.encoder
        length = input_ids.shape[1]
        mask = (1.0 - attention_mask[:, None, None, :].to(encoder.dtype)).expand(-1, 1, length, -1)
        mask = mask.masked_fill(mask.bool(), torch.finfo(encoder.dtype).min)
        local = mask.masked_fill(~self.local_allowed[:length, :length], torch.finfo(encoder.dtype).min)
        h = encoder.embeddings(input_ids=input_ids)
        for layer in encoder.layers:
            h = layer(h, attention_mask=mask, sliding_window_mask=local,
                      position_ids=self.positions[:, :length])[0]
        h = encoder.final_norm(h)
        return self.model.score_hidden(h, attention_mask, marker_pos, marker_mask, qtype)


class CudaGraphRunner:
    def __init__(self, model, device, dtype, config, max_len=1024):
        self.model, self.device, self.dtype = model, device, dtype
        self.prepared = PreparedModel(model, max_len).eval()
        self.max_len = max_len
        self.capacity = config.graph_cache_size
        self.graphs = OrderedDict()
        self.counters = Counter()
        self.capture_ms = 0.0
        self.stream_count = config.graph_streams
        self.raw_observer = None
        self.compiled = config.runner == "cuda-graph-compile"
        self.compile_capacity = config.compile_cache_size
        self.compiled_shapes, self.compile_fallback_shapes = set(), set()
        self.function = self.prepared
        if self.compiled:
            # 串行子序列仅按 length/marker count 专化，且有显式容量；不会无限编译。
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, self.compile_capacity * 2 + 8
            )
            self.function = torch.compile(
                self.prepared, fullgraph=True, dynamic=False,
                options={"emulate_precision_casts": True, "triton.cudagraphs": False},
            )
        self.info = {"attention_backend": "TORCH_SDPA", "head_compilation": self.compiled,
                     "compilation_mode": "INDUCTOR_FULL_MODEL" if self.compiled else "NONE",
                     "compile_profile_capacity": self.compile_capacity if self.compiled else 0,
                     "cudagraph_mode": "FULL_MODEL", "graph_cache_capacity": self.capacity,
                     "graph_shape_policy": "exact_sequence_power_of_two_batch",
                     "graph_streams": self.stream_count,
                     "batch_execution": "independent_sequences",
                     "graph_prewarm_profiles": config.graph_prewarm_profiles,
                     "emulate_eager_rounding": True}
        for profile in config.graph_prewarm_profiles:
            self.prewarm(*profile)

    def prewarm(self, count, length, options):
        if length > self.max_len:
            raise ValueError("graph_prewarm_sequence_limit_exceeded")
        batch = {
            "input_ids": torch.zeros((count, length), dtype=torch.long),
            "attention_mask": torch.ones((count, length), dtype=torch.long),
            "marker_pos": torch.arange(options).expand(count, -1).clone(),
            "marker_mask": torch.ones((count, options), dtype=torch.bool),
            "qtype": torch.zeros(count, dtype=torch.long),
        }
        self.execute(batch)

    def forward(self, tensors):
        function = self.function
        shape = (tensors[0].shape[1], tensors[2].shape[1])
        if self.compiled:
            if shape in self.compiled_shapes or len(self.compiled_shapes) < self.compile_capacity:
                self.compiled_shapes.add(shape)
            else:
                function = self.prepared
                self.compile_fallback_shapes.add(shape)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=self.dtype in (torch.float16, torch.bfloat16)):
            return function(*tensors)

    def sequences(self, tensors, root=None, branches=()):
        if root is not None:
            for branch in branches:
                branch.wait_stream(root)
        values = []
        for row in range(tensors[0].shape[0]):
            if branches:
                with torch.cuda.stream(branches[row % len(branches)]):
                    values.append(self.forward([tensor[row:row + 1] for tensor in tensors]))
            else:
                values.append(self.forward([tensor[row:row + 1] for tensor in tensors]))
        if root is not None:
            for branch in branches:
                root.wait_stream(branch)
        logits, acts = (torch.cat([value[i] for value in values]) for i in range(2))
        return logits.float(), acts.float(), torch.softmax(acts.float(), -1)

    @torch.inference_mode()
    def capture(self, tensors):
        started = perf_counter()
        static = [tensor.clone() for tensor in tensors]
        stream = torch.cuda.Stream(device=self.device)
        branches = [torch.cuda.Stream(device=self.device)
                    for _ in range(min(self.stream_count, static[0].shape[0]))]
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.sequences(static, stream, branches)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self.sequences(static, stream, branches)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.capture_ms += (perf_counter() - started) * 1000
        self.counters["captures"] += 1
        return graph, static, outputs, stream, branches

    @torch.inference_mode()
    def execute(self, batch):
        if not batch["attention_mask"].all() or not batch["marker_mask"].all():
            raise ValueError("cuda_graph_requires_exact_sequence_and_option_shapes")
        if batch["input_ids"].shape[1] > self.max_len:
            raise ValueError("cuda_graph_sequence_limit_exceeded")
        tensors = [batch[key].to(self.device) for key in INPUT_KEYS]
        count = tensors[0].shape[0]
        padded = 1 << (count - 1).bit_length()
        if count < padded:
            tensors = [torch.cat((tensor, tensor[:1].expand(padded - count, *tensor.shape[1:])))
                       for tensor in tensors]
        shape = (*tensors[0].shape, tensors[2].shape[1])
        if shape not in self.graphs:
            if len(self.graphs) == self.capacity:
                _, (expired, _, _, _, _) = self.graphs.popitem(last=False)
                expired.reset()
                self.counters["evictions"] += 1
            self.graphs[shape] = self.capture(tensors)
        self.graphs.move_to_end(shape)
        graph, static, outputs, _, _ = self.graphs[shape]
        for target, source in zip(static, tensors):
            target.copy_(source)
        graph.replay()
        self.counters["graph_replays"] += 1
        self.counters["input_sequences"] += count
        self.counters["padded_sequences"] += padded
        if self.raw_observer is not None:
            self.raw_observer(batch, outputs[0][:count], outputs[1][:count])
        # 拷贝后才复用静态输出；GPU 的同一线程／stream 独占这一 runner。
        return outputs[0][:count].cpu().numpy(), outputs[2][:count].cpu().numpy()

    def metrics(self):
        return {"counters": dict(self.counters), "capture_ms": self.capture_ms,
                "profiles": [list(shape) for shape in self.graphs],
                "cache_size": len(self.graphs), "cache_capacity": self.capacity,
                "compiled_shapes": [list(shape) for shape in sorted(self.compiled_shapes)],
                "compile_fallback_shapes": [list(shape) for shape in sorted(self.compile_fallback_shapes)]}

    def close(self):
        self.graphs.clear()
