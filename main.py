import numpy as np
np.random.seed(42)

from pairs import list_pairs, get_pair
from walk_forward import run_walk_forward, print_report


def print_summary(results):
    # cross-pair comparison: condense the key metrics of each pair's aggregate
    # into a single table so reviewers can scan the Study 1 narrative quickly
    print(f'\n\n{"="*78}')
    print('Study 1 Cross-Pair Summary')
    print(f'{"="*78}')

    print(f"\n{'pair':<14}{'folds':<9}{'total_ret':<12}"
          f"{'sharpe':<10}{'beta_m':<10}{'beta_pv':<10}{'r_sq':<8}")
    print('-' * 78)

    for key, result in results.items():
        pair = get_pair(key)
        agg = result['aggregate']
        folds_str = f"{result['n_traded']}/{result['n_total']}"

        if agg is None:
            print(f"{pair['label']:<14}{folds_str:<9}"
                  f"{'n/a':<12}{'n/a':<10}{'n/a':<10}{'n/a':<10}{'n/a':<8}")
        else:
            print(f"{pair['label']:<14}{folds_str:<9}"
                  f"{agg['total_return']:<12.4f}"
                  f"{agg['sharpe']:<10.3f}"
                  f"{agg['beta_m']:<10.4f}"
                  f"{agg['beta_pvalue']:<10.3f}"
                  f"{agg['r_squared']:<8.4f}")


if __name__ == '__main__':
    keys = list_pairs(study=1)
    results = {}

    for key in keys:
        pair = get_pair(key)
        print(f"\n>>> Running Walk-Forward on {pair['label']}")
        results[key] = run_walk_forward(pair, '2016-01-01', '2026-01-01')
        print_report(results[key], pair)

    print_summary(results)