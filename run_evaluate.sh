echo ====================================================
echo ============ Results for testing files ============
echo TEST Results for en-de and en-zh
python main.py -m predict -d test -f en-*/*.tsv
python main.py -m evaluate -d test -f en-*/*.tsv

echo ====================================================
echo TEST Results for en-de
python main.py -m predict -d test -f en-de/*.tsv
python main.py -m evaluate -d test -f en-de/*.tsv

echo ====================================================
echo TEST Results for en-zh
python main.py -m predict -d test -f en-zh/*.tsv
python main.py -m evaluate -d test -f en-zh/*.tsv
echo ============ Results for testing files ============ 
echo ====================================================
echo 
echo ======================================================
echo ============ Results for validation files ============ 
echo VALID Results for et-en, ne-en and ro-en
python main.py -m predict -d valid -f *-en/*.tsv
python main.py -m evaluate -d valid -f *-en/*.tsv

echo ======================================================
echo VALID Results for et-en
python main.py -m predict -d valid -f et-en/*.tsv
python main.py -m evaluate -d valid -f et-en/*.tsv

echo ======================================================
echo VALID Results for ne-en
python main.py -m predict -d valid -f ne-en/*.tsv
python main.py -m evaluate -d valid -f ne-en/*.tsv

echo ======================================================
echo VALID Results for ro-en
python main.py -m predict -d valid -f ro-en/*.tsv
python main.py -m evaluate -d valid -f ro-en/*.tsv
echo ============ Results for validation files ============ 
echo ======================================================


