from argparse import ArgumentParser
from time import time

from joblib import Parallel, delayed
from tqdm import tqdm
import torch

from src.dna import GenePool
from src.cpu.env import Env

MAX_GAME_LEN = 5000
POPULATION = 100
GAME_SIZE = 100
WIN_BONUS = 1000
DEVICE = 'cpu' # Why is this faster than my GPU??

def generation(gene_pool: GenePool):
    st = time()
    blues,reds = torch.randperm(gene_pool.population).chunk(2)
    n_games = blues.size(0)
    n_winners = gene_pool.population // 2

    print("\tSimulating... ", end='', flush=True)
    def game(g):
        b_genes, b_sexes = gene_pool.create_swarm(GAME_SIZE, blues[g:g+1])
        r_genes, r_sexes = gene_pool.create_swarm(GAME_SIZE, reds[g:g+1])

        env = Env(
            b_genes.squeeze(0), b_sexes.squeeze(0),
            r_genes.squeeze(0), r_sexes.squeeze(0)
        )

        game_over = torch.zeros(2)

        for _ in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        draw = game_over.sum() == 0 or game_over.sum() == 2

        game_len = MAX_GAME_LEN
        b_score = 0
        r_score = 0
        if not draw and game_over[1]:
            # Team bonuses for winning quickly
            game_len = env.b_alive_time.max()
            speed_bonus = (MAX_GAME_LEN - game_len) / MAX_GAME_LEN
            speed_bonus *= (WIN_BONUS / 2)
            b_score += WIN_BONUS + speed_bonus


        elif not draw and game_over[0]:
            # Team bonuses for winning quickly
            game_len = env.r_alive_time.max()
            speed_bonus = (MAX_GAME_LEN - game_len) / MAX_GAME_LEN
            speed_bonus *= (WIN_BONUS / 2)
            r_score += WIN_BONUS + speed_bonus

        return b_score, r_score

    scores = Parallel(n_games, prefer='processes')(
        delayed(game)(g) for g in range(0, n_games)
    )
    en = time()

    print(f" ({en-st:0.2f}s)")

    b_scores, r_scores = zip(*scores)
    b_scores = torch.tensor(b_scores, dtype=torch.float32)
    r_scores = torch.tensor(r_scores, dtype=torch.float32)

    ordered_scores = torch.zeros(gene_pool.population, dtype=torch.float32)
    ordered_scores[blues] = b_scores
    ordered_scores[reds] = r_scores

    winners = ordered_scores.sort(descending=True).indices[:n_winners]
    gene_pool.reproduce(winners)


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
    best = MAX_GAME_LEN

    for e in range(1,100_000):
        generation(pool)

        scores = evaluate(pool)
        avg = MAX_GAME_LEN - (sum(scores) / len(scores))
        max_v = MAX_GAME_LEN - min(scores)

        print(f'[{e}] Avg: {int(avg)}, Best: {max_v}', end='')
        if avg > best:
            best = avg
            print('*')
            pool.save('genes/best.pt')
        else:
            print()

        if e % 100 == 0:
            pool.save(f'genes/{e // 100}.pt')

        with open(log, 'a') as f:
            f.write(f'{e},{avg},{max_v}\n')

        pool.save('genes/current.pt')

if __name__ == '__main__':
    with open('log.txt', 'w+') as f:
        f.write('generation,avg,best\n')
    train()