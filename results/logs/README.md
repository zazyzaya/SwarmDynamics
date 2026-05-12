# V2

Changes:

* Added obstacles
* Added gene to avoid obstacles
* Corrected physics glitch to cap rotational velocity
* Using annealed curriculum learning to always have to fight some default drones

Defaults:

    ap.add_argument('--game-size', default=100, type=int)
    ap.add_argument('--population', default=100, type=int)
    ap.add_argument('--cull-rate', default=2, type=int)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--gene-init', default='hybrid', choices=['hybrid', 'random', 'baseline'])
    ap.add_argument('--xover-rate', default=0.75, type=float)
    ap.add_argument('--mute-rate', default=0.05, type=float)
    ap.add_argument('--mute-stren', default=0.25, type=float)
    ap.add_argument('--xover-alpha', default=0.1, type=float)
    ap.add_argument('--save-rate', default=100, type=int)
    ap.add_argument('--eval-rate', default=10, type=int)
    ap.add_argument('--num-obstacles', default=10, type=int)
    ap.add_argument('--min-baseline-games', default=0.15, type=float)