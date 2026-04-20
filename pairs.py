from data import load_pair


# central registry of all pairs traded across Study 1 and Study 2
pairs = {
    'KOPEP': {
        'ticker_y': 'KO', 'ticker_x': 'PEP',
        'label': 'KO / PEP', 'study': 1,
    },
    'GSMS': {
        'ticker_y': 'GS', 'ticker_x': 'MS',
        'label': 'GS / MS', 'study': 1,
    },
    'NVDAMSFT': {
        'ticker_y': 'NVDA', 'ticker_x': 'MSFT',
        'label': 'NVDA / MSFT', 'study': 2,
    },
}


def get_pair(key):
    # dict lookup with a helpful error message
    if key not in pairs:
        raise KeyError(
            f"pair '{key}' not found. available keys: {list(pairs.keys())}"
        )
    return pairs[key]


def list_pairs(study=None):
    # return all pair keys, optionally filtered by study number
    if study is None:
        return list(pairs.keys())
    return [k for k, v in pairs.items() if v['study'] == study]


def load_smoke(key, start='2016-01-01', end='2026-01-01'):
    # convenience unpacker used by every module's smoke test
    pair = get_pair(key)
    prices = load_pair(pair['ticker_y'], pair['ticker_x'], start, end)
    y = prices[pair['ticker_y']]
    x = prices[pair['ticker_x']]
    return prices, y, x, pair
