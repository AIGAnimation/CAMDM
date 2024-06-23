import copy
import json
from types import SimpleNamespace as Namespace

def add_model_args(parser):
    parser.add_argument('--rot_req', type=str, help='Rotation representation: choose from "q", "euler", or "6d".')
    parser.add_argument('--decoder', type=str, help='Type of decoder to use.')
    parser.add_argument('--latent_dim', type=int, help='Width of the Transformer/GRU layers.')
    parser.add_argument('--ff_size', type=int, help='Feed-forward size for Transformer/GRU.')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, help='Number of layers in the model.')
    parser.add_argument('--offset_frame', type=float, help='The number of frames for a training clip. Use 0 for a random number.')
    parser.add_argument('--past_frame', type=int, help='The number of past frames for a training clip. Use 0 for a random number.')
    parser.add_argument('--future_frame', type=int, help='The number of future frames for a training clip. Use 0 for a random number.')
    parser.add_argument('--local_cond', default=None, type=str, help='Local conditioning.')
    parser.add_argument('--global_cond', default=None, type=str, help='Global conditioning.')


def add_diffusion_args(parser):
    parser.add_argument('--noise_schedule', type=str, help='Noise schedule: choose from "cosine", "linear", or "linear1".')
    parser.add_argument('--diffusion_steps', type=int, help='Number of diffusion steps.')
    parser.add_argument('--sigma_small', type=bool, help='Use small sigma values.')


def add_train_args(parser):
    parser.add_argument('--epoch', type=int, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, help='Learning rate for training.')
    parser.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of steps to anneal the learning rate.")
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for the optimizer.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--loss_terms', type=str, help='Loss terms to use in training. Format: [mse_rotation, positional_loss, velocity_loss, foot_contact]. Use 0 for No, 1 for Yes, e.g., "1111".')
    parser.add_argument('--cond_mask_prob', type=float, help='Probability of masking conditioning.')
    parser.add_argument('--ema', default=False, type=bool, help='Use Exponential Moving Average (EMA) for model parameters.')
    parser.add_argument('--workers', default=8, type=int, help='Number of workers for data loading.')


def config_parse(args):
    config = copy.deepcopy(json.load(open(args.config), object_hook=lambda d: Namespace(**d)))

    config.data = args.data
    config.name = args.name    

    config.arch.rot_req = str(args.rot_req) if args.rot_req is not None else config.arch.rot_req
    config.arch.decoder = str(args.decoder) if args.decoder is not None else config.arch.decoder
    config.arch.latent_dim = int(args.latent_dim) if args.latent_dim is not None else config.arch.latent_dim
    config.arch.ff_size = int(args.ff_size) if args.ff_size is not None else config.arch.ff_size
    config.arch.num_heads = int(args.num_heads) if args.num_heads is not None else config.arch.num_heads
    config.arch.num_layers = int(args.num_layers) if args.num_layers is not None else config.arch.num_layers
    config.arch.offset_frame = int(args.offset_frame) if args.offset_frame is not None else config.arch.offset_frame
    config.arch.past_frame = int(args.past_frame) if args.past_frame is not None else config.arch.past_frame
    config.arch.future_frame = int(args.future_frame) if args.future_frame is not None else config.arch.future_frame
    config.arch.clip_len = config.arch.past_frame + config.arch.future_frame
    config.arch.local_cond = str(args.local_cond) if args.local_cond is not None else config.arch.local_cond
    config.arch.global_cond = str(args.global_cond) if args.global_cond is not None else config.arch.global_cond
        
    config.diff.noise_schedule = str(args.noise_schedule) if args.noise_schedule is not None else config.diff.noise_schedule
    config.diff.diffusion_steps = int(args.diffusion_steps) if args.diffusion_steps is not None else config.diff.diffusion_steps
    config.diff.sigma_small = True if args.sigma_small else config.diff.sigma_small

    config.trainer.epoch = int(args.epoch) if args.epoch is not None else config.trainer.epoch
    config.trainer.lr = float(args.lr) if args.lr is not None else config.trainer.lr
    config.trainer.weight_decay = args.weight_decay
    config.trainer.lr_anneal_steps = args.lr_anneal_steps
    config.trainer.cond_mask_prob = args.cond_mask_prob if args.cond_mask_prob is not None else config.trainer.cond_mask_prob
    config.trainer.batch_size = int(args.batch_size) if args.batch_size is not None else config.trainer.batch_size
    config.trainer.ema = True #if args.ema else config.trainer.ema
    config.trainer.workers = int(args.workers) 
    loss_terms = args.loss_terms if args.loss_terms is not None else config.trainer.loss_terms
    config.trainer.use_loss_mse = True if loss_terms[0] == '1' else False
    config.trainer.use_loss_3d = True if loss_terms[1] == '1' else False
    config.trainer.use_loss_vel = True if loss_terms[2] == '1' else False
    config.trainer.use_loss_contact = True if loss_terms[3] == '1' else False
    config.trainer.load_num = -1
    config.trainer.save_freq = int(config.trainer.epoch // 10)

    data_prefix = args.data.split('/')[-1].split('.')[0]
    config.save = '%s/%s_%s' % (args.save, args.name, data_prefix) if 'debug' not in config.name else '%s/%s' % (args.save, args.name)
    return config