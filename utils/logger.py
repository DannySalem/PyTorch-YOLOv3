from torch.utils.tensorboard import SummaryWriter
import time


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        self.writer.add_scalar(tag, value, global_step=step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        loss_metrics = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', ]
        validation_metrics = ['precision', 'recall', 'mAP', 'f1']

        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        # self.writer.add_summary(summary, step)
        # self.writer.add_scalars('Summary', dict(tag_value_pairs), global_step=step)
        synced_time = time.time()
        for key, value in tag_value_pairs:
            if key == 'loss':
                self.writer.add_scalars('Loss Metrics/Total Loss', {key: value}, global_step=step, walltime=synced_time)
            else:
                prefix, suffix = key.rsplit('_', 1)
                if prefix in loss_metrics:
                    self.writer.add_scalars(f'Loss Metrics/{prefix}', {key: value}, global_step=step, walltime=synced_time)
                elif suffix in validation_metrics:
                    self.writer.add_scalars(f'Validation Metrics/{suffix}', {key: value}, global_step=step, walltime=synced_time)
                else:
                    self.writer.add_scalars(f'Acccuracy Metrics/{prefix}', {key: value}, global_step=step, walltime=synced_time)

        # for x in tag_value_pairs:
        #    self.writer.add_scalar(x[0], x[1], global_step=step)
