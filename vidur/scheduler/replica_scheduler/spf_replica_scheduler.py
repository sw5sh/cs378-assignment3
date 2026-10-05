from vidur.entities.batch import Request
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)


class SpfReplicaScheduler(SarathiReplicaScheduler):
    """Sarathi batching (Part 3), but waiting requests start shortest prompt
    first, with aging (Part 4).

    Before each iteration, Sarathi's _get_next_batch calls
    _order_waiting_queue(), so _add_new_requests starts requests from the
    front of the reordered queue.
    """

    def _priority(self, request: Request, now: float) -> float:
        """Smaller starts first.

            priority = prompt tokens left - aging * seconds waited

        prompt tokens left: request.num_prefill_tokens -
            request.num_processed_prefill_tokens
        aging:              self._config.aging_tokens_per_s
        seconds waited:     now - request.arrived_at

        With aging 0 this is plain shortest-prompt-first. With aging > 0 a
        waiting request gains `aging` tokens of priority every second, so a
        long prompt eventually overtakes newer short ones.
        """
        # [Part 4] TODO: return the priority above.
        raise NotImplementedError("Part 4: fill _priority")
        # END OF YOUR CODE

    def _order_waiting_queue(self) -> None:
        """Sort self._request_queue in place by _priority(request, self._now),
        smallest first. Requests with equal priority keep their current order.
        Do not drop or copy requests.
        """
        # [Part 4] TODO: sort the waiting queue.
        raise NotImplementedError("Part 4: fill _order_waiting_queue")
        # END OF YOUR CODE
