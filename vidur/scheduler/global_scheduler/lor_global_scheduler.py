from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):
    """Least outstanding requests (LOR): send each request to the replica with
    the fewest requests in flight.

    A request is in flight on a replica from when it is sent there until it
    finishes: waiting in the replica's queue or running.
    self.outstanding_requests(replica_id) returns those requests.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        """Assign every request in self._request_queue (oldest first) to a
        replica. Return a list of (replica_id, request) pairs.

        For each request, pick the replica with the fewest outstanding
        requests; on a tie, the lowest replica_id. Then count the request on
        that replica right away, so later requests in the same call see it.
        """
        self.sort_requests()
        request_mapping = []
        # [Part 5] TODO: count each replica's outstanding requests, then
        # assign every request in self._request_queue (pop from the front).
        # Replica ids: self._replica_schedulers.keys().
        raise NotImplementedError("Part 5: fill LOR schedule")
        # END OF YOUR CODE
        return request_mapping
