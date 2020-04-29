# Both zh and de
python main.py -m ensemble_train -d valid -f */*.tsv

python main.py -m ensemble_predict -d test -f */*.tsv

python main.py -m evaluate -d test -f */*.tsv

# Separate
# python main.py -m ensemble_predict -d test -f en-zh/*.tsv

# python main.py -m evaluate -d test -f en-zh/*.tsv