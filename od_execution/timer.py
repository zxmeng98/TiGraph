import torch

class CudaEventTimer(object):
    def __init__(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        torch.cuda.current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class Timer:
    """Timer."""
    def __init__(self):
        self.started_ = False
        self.event_timers = []
        self.start_event = None
        self.elapsed_records = None

    def start(self):
        """Start the timer."""
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        self.started_ = True

    def stop(self, reset=False, record=False):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self.event_timers.append(CudaEventTimer(self.start_event, end_event))
        self.start_event = None
        self.started_ = False

    def _get_elapsed_msec(self):
        self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
        self.event_timers.clear()
        return sum(self.elapsed_records)

    def reset(self):
        """Reset timer."""
        self.started_ = False
        self.start_event = None
        self.elapsed_records = None
        self.event_timers.clear()

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self._get_elapsed_msec()
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_
