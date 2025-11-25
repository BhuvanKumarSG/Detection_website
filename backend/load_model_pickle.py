import sys
from pathlib import Path
import traceback

def try_joblib(path):
    try:
        import joblib
        obj = joblib.load(path)
        return obj
    except Exception as e:
        return e

def try_pickle(path):
    try:
        import pickle
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        return e

def inspect(obj):
    print('Loaded object type:', type(obj))
    try:
        print('repr(obj)[:1000]:', repr(obj)[:1000])
    except Exception:
        pass
    # list some attributes
    attrs = dir(obj)
    print('\nSome attributes:')
    for a in attrs[:60]:
        print(' ', a)
    print('\nHas predict?:', hasattr(obj, 'predict'))
    print('Has predict_proba?:', hasattr(obj, 'predict_proba'))
    print('Has score?:', hasattr(obj, 'score'))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python load_model_pickle.py <path>')
        raise SystemExit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print('File not found:', path); raise SystemExit(2)

    print('Trying joblib.load...')
    r = try_joblib(path)
    if not isinstance(r, Exception):
        print('joblib.load succeeded')
        inspect(r)
        raise SystemExit(0)
    else:
        print('joblib.load failed:', r)

    print('\nTrying pickle.load...')
    r2 = try_pickle(path)
    if not isinstance(r2, Exception):
        print('pickle.load succeeded')
        inspect(r2)
        raise SystemExit(0)
    else:
        print('pickle.load failed:', r2)
        traceback.print_exception(type(r2), r2, r2.__traceback__)
        raise SystemExit(3)
