from argparse import ArgumentParser

from src.dna import GenePool

LOG_FILE = ''
TAG = ''

def train(hyperparams):
    params = dict(
        device=hyperparams.device,
        xover_rate=hyperparams.xover_rate,
        xover_alpha=hyperparams.xover_alpha,
        mute_rate=hyperparams.mute_rate,
        mute_stren=hyperparams.mute_stren,
        hybrid_init=hyperparams.gene_init == 'hybrid',
        use_baseline=hyperparams.gene_init == 'baseline'
    )
    pool = GenePool(
        hyperparams.population,
        **params
    )

    best = 0
    for e in range(100_000):
        stats = generation(
            pool, e,
            hyperparams.cull_rate,
            hyperparams.game_size
        )

        if e % hyperparams.eval_rate == 0:
            scores, eval_t = evaluate(pool, hyperparams.game_size)

            avg = sum(scores) / len(scores)
            max_v = max(scores)

            print(f'\tAvg: {avg:0.4f}, Best: {max_v:0.4f}', end='')
            if avg > best:
                best = avg
                print('*')
                pool.save(f'genes/best{TAG}.pt')
            else:
                print()

        else:
            avg = ''; max_v = ''; eval_t=0

        with open(LOG_FILE, 'a') as f:
            f.write(f'{e},{avg},{max_v},{",".join([str(s) for s in stats])},{eval_t}\n')

        if e % hyperparams.save_rate == 0:
            pool.save(f'genes/{e // 100}{TAG}.pt')

        pool.save(f'genes/current{TAG}.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
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
    ap.add_argument('--tag')

    args = ap.parse_args()

    if args.device == 'cpu':
        from src.cpu.evolve_cpu import generation, evaluate
    else:
        args.device = int(args.device)
        from src.cuda.evolve_gpu import generation, evaluate

    LOG_FILE = 'results/logs/'
    if (tag := args.tag):
        LOG_FILE += tag + '.txt'
        TAG = f'-{tag}'
    else:
        LOG_FILE += 'log.txt'
        TAG = ''

    with open(LOG_FILE, 'w+') as f:
        f.write('generation,eval_avg,eval_best,avg_fitness,topk_fitness,avg_fitness_std,topk_fitness_std,avg_len,tr_time,eval_time\n')

    train(args)