from argparse import ArgumentParser
from time import time

from joblib import Parallel, delayed
from tqdm import tqdm
import torch

from src.dna import GenePool, MAX_GAME_LEN
from src.cpu.env import Env

POPULATION = 100
GAME_SIZE = 100
WIN_BONUS = 1000
DEVICE = 'cpu' # Why is this faster than my GPU??

def generation(gene_pool: GenePool, e):
    st = time()
    blues,reds = torch.randperm(gene_pool.population).chunk(2)
    n_games = blues.size(0)
    n_winners = gene_pool.population // 2

    def game(g):
        b_genes, b_sexes = gene_pool.create_swarm(GAME_SIZE, blues[g:g+1])
        r_genes, r_sexes = gene_pool.create_swarm(GAME_SIZE, reds[g:g+1])

        env = Env(
            b_genes.squeeze(0), b_sexes.squeeze(0),
            r_genes.squeeze(0), r_sexes.squeeze(0)
        )

        game_over = torch.zeros(2)

        for steps in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        b_kills = env.b_kills.sum() / GAME_SIZE
        r_kills = env.r_kills.sum() / GAME_SIZE
        game_len = max(env.b_alive_time.max(), env.r_alive_time.max())

        b_score, r_score = GenePool.fitness(b_kills, r_kills, game_len)
        return b_score, r_score, steps

    scores = Parallel(n_games, prefer='processes')(
        delayed(game)(g) for g in range(0, n_games)
    )
    en = time()

    print(f" ({en-st:0.2f}s)")

    b_scores, r_scores, steps = zip(*scores)
    b_scores = torch.tensor(b_scores, dtype=torch.float32)
    r_scores = torch.tensor(r_scores, dtype=torch.float32)

    ordered_scores = torch.zeros(gene_pool.population, dtype=torch.float32)
    ordered_scores[blues] = b_scores
    ordered_scores[reds] = r_scores

    avg_len = sum(steps) / len(steps)
    avg_fitness = ordered_scores.mean().item()
    avg_fitness_std = ordered_scores.std().item()

    winners = ordered_scores.topk(n_winners)
    top_fitness = winners.values.mean().item()
    top_fitness_std = winners.values.std().item()
    print(
        f"[{e}] Steps: {int(avg_len)},",
        f"Avg fitness: {avg_fitness:0.4f}+/-{avg_fitness_std:0.2f},",
        f"Top fitness: {top_fitness:0.4f}+/-{top_fitness_std:0.2f}",
        f"({en-st:0.2f}s)"
    )


    gene_pool.reproduce(winners.indices)

    return avg_fitness, top_fitness, avg_fitness_std, top_fitness_std, avg_len


def evaluate(gene_pool: GenePool):
    st = time()
    default = GenePool(1, device=DEVICE, use_baseline=True)
    n_games = gene_pool.population

    print("\tEvaluating... ", end='', flush=True)
    def game(g):
        blue_queen = torch.tensor([g], device=DEVICE)
        b_genes, b_sexes = gene_pool.create_swarm(GAME_SIZE, blue_queen)

        r_genes, r_sexes = default.create_swarm(GAME_SIZE, torch.tensor([0]))

        env = Env(
            b_genes.squeeze(0), b_sexes.squeeze(0),
            r_genes.squeeze(0), r_sexes.squeeze(0)
        )

        game_over = torch.zeros(2)
        for step in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        # Not loss or draw
        if game_over[1] and not game_over[0]:
            return step
        else:
            return MAX_GAME_LEN

    steps_to_win = Parallel(n_games, prefer='processes')(
        delayed(game)(g) for g in range(n_games)
    )
    en = time()
    print(f'({en-st:0.2f}s)')

    return steps_to_win

def train():
    pool = GenePool(POPULATION, device=DEVICE, hybrid_init=True)

    log = 'log.txt'
    best = 0

    for e in range(100_000):
        stats = generation(pool, e)

        if e % 10 == 0:
            scores = evaluate(pool)

            avg = (MAX_GAME_LEN - (sum(scores) / len(scores))) / MAX_GAME_LEN
            max_v = (MAX_GAME_LEN - min(scores)) / MAX_GAME_LEN

            print(f'\tAvg: {avg:0.4f}, Best: {max_v:0.4f}', end='')
            if avg > best:
                best = avg
                print('*')
                pool.save('genes/best.pt')
            else:
                print()

        else:
            avg = ''; max_v = ''

        with open(log, 'a') as f:
            f.write(f'{e},{avg},{max_v},{",".join([str(s) for s in stats])}\n')

        if e % 100 == 0:
            pool.save(f'genes/{e // 100}.pt')

        pool.save('genes/current.pt')

if __name__ == '__main__':
    with open('log.txt', 'w+') as f:
        f.write('generation,eval_avg,eval_best,avg_fitness,topk_fitness,avg_fitness_std,topk_fitness_std,avg_len\n')
    train()