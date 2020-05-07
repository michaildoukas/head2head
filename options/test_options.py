from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--time_fwd_pass', action='store_true', help='Show the forward pass time for synthesizing each frame.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--do_reenactment', action='store_true', default=False, help='When set, perform source to target head reenactment and not self-reenactment.')
        self.isTrain = False
