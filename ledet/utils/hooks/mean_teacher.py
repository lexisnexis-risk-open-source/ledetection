from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n


@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warmup=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warmup = warmup
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        # Only do it once at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of Student to Teacher...")
            self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update EMA parameters every self.interval iterations."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.curr_step = runner.iter
        # Add an attribute in the model to track current step for downstream purposes.
        model.curr_step = self.curr_step
        if self.curr_step % self.interval != 0:
            return
        # We warm up the momentum considering the instability at the beginning.
        momentum = min(
            self.momentum, 1 - (1 + self.warmup) / (self.curr_step + 1 + self.warmup)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, self.curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
