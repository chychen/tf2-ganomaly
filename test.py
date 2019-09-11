from absl import app
from absl import flags
from absl import logging
import os
FLAGS = flags.FLAGS
# flags.DEFINE_string("name", None, "Your name.")
# flags.DEFINE_string("log_dir", 'test.log',
#                      "Number of times to print greeting.")
flags.DEFINE_integer("num_times", 1, "Number of times to print greeting.")


def main(_):
    if not os.path.exists('aa'):
        os.makedirs('aa')
    handler = logging.get_absl_handler()
    handler.use_absl_log_file('aaa', log_dir='aa')
    logging.info('aa')


if __name__ == '__main__':
    app.run(main)
