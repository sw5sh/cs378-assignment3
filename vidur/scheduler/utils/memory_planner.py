from vidur.config import ReplicaConfig
from vidur.entities.replica import Replica
from vidur.utils.param_counter import ParamCounter


class MemoryPlanner:
    def __init__(self, replica_config: ReplicaConfig, replica: Replica) -> None:
        self._param_counter = ParamCounter(replica_config)
        self._replica = replica

    def _get_kv_cache_memory_per_layer_per_request(self) -> int:
        # [Warmup] TODO: KV-cache bytes for ONE request, ONE layer.
        # HINT: fp16 is 2 bytes; store both K and V; use
        #   self._replica.attention_head_dim
        #   self._replica.kv_heads_per_tensor_parallel_worker
        #   self._replica.max_request_tokens
        raise NotImplementedError("Warmup: fill _get_kv_cache_memory_per_layer_per_request")

        # END OF YOUR CODE

    def _get_parameter_memory_per_device(self) -> int:
        # [Warmup] TODO: model-parameter bytes on this device (fp16).
        # HINT: 2 * self._param_counter.get_num_parameters_per_device()
        raise NotImplementedError("Warmup: fill _get_parameter_memory_per_device")

        # END OF YOUR CODE

    def _get_kv_cache_memory_per_device_per_request(self) -> int:
        # [Warmup] TODO: KV-cache bytes for ONE request across ALL layers on
        # this device.
        # HINT: per-layer KV * self._replica.num_layers
        raise NotImplementedError(
            "Warmup: fill _get_kv_cache_memory_per_device_per_request"
        )

        # END OF YOUR CODE

    def get_max_batch_size(self) -> int:
        # [Warmup] TODO: how many concurrent requests fit in leftover GPU memory?
        # HINT:
        #   available = total_memory_gb * 1024**3 * (1 - memory_margin_fraction)
        #   leftover  = available - parameter_memory_per_device
        #   then integer-divide by kv_cache_memory_per_device_per_request.
        #   Assert that at least one request fits, then return that count.
        # Use:
        #   self._replica.total_memory_gb
        #   self._replica.memory_margin_fraction
        #   self._get_parameter_memory_per_device()
        #   self._get_kv_cache_memory_per_device_per_request()
        raise NotImplementedError("Warmup: fill get_max_batch_size")

        # END OF YOUR CODE

    def get_max_request_slots(self) -> int:
        # [Warmup] TODO: how many request slots across the pipeline?
        # HINT: get_max_batch_size() * self._replica.num_pipeline_stages
        raise NotImplementedError("Warmup: fill get_max_request_slots")

        # END OF YOUR CODE
