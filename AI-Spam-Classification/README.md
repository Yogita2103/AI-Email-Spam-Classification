# AI Email Spam Classifier

A machine-learning web app that reads an email and tells you whether it is **Spam** or **Not Spam**.
It is built with Python, scikit-learn (TF-IDF + Multinomial Naive Bayes) and Flask, with an HTML/CSS/JavaScript front end.

## Features
- Paste an email and get a verdict with spam probability and model confidence
- Shared text-cleaning function used in both training and prediction
- Evaluation page section: accuracy, precision, recall, F1-score and confusion matrix
- JSON API (`/predict`, `/metrics`, `/health`)
- Automated tests with `pytest`
- Ready for GitHub and Render deployment

## Dataset
[SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/) (real emails, about 6,000 messages, roughly 31% spam).
Limitation: the emails date from 2002-2005, so modern spam styles may be missed.

## How it works
1. `src/build_dataset.py` reads the raw emails and writes `data/emails.csv`
2. `src/preprocess.py` cleans text (lowercase, strip HTML, mark URLs/addresses/numbers, remove punctuation)
3. `src/train.py` builds a `TfidfVectorizer` + `MultinomialNB` pipeline, evaluates it on a held-out 20% test set, and saves the model, `metrics.json` and the confusion matrix image
4. `app.py` loads the saved model and serves the website and API

## Project structure
```
AI-Spam-Classification/
├── app.py                 Flask app
├── requirements.txt       Pinned dependencies
├── render.yaml            Render deployment config
├── .python-version        Python version for Render
├── data/                  raw/ (not pushed) and emails.csv
├── src/                   build_dataset.py, preprocess.py, train.py, predict.py
├── model/                 spam_pipeline.joblib, metrics.json (generated)
├── templates/index.html   Web page
├── static/                css/, js/, img/confusion_matrix.png
└── tests/                 pytest tests
```

## Run it locally

Activate your virtual environment first (`(venv)` should show in the terminal).

**1. Check your library versions** (the numbers in `requirements.txt` must match these):
```
python -c "import sys, numpy, pandas, sklearn, matplotlib, flask; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__); print('matplotlib', matplotlib.__version__); print('flask', flask.__version__)"
```

**2. Download the dataset** from https://spamassassin.apache.org/old/publiccorpus/ into `data/raw/`:
`20030228_easy_ham.tar.bz2`, `20030228_easy_ham_2.tar.bz2`, `20030228_hard_ham.tar.bz2`, `20030228_spam.tar.bz2`, `20030228_spam_2.tar.bz2`

**3. Extract them** (Windows 10/11, Mac and Linux all have `tar`):
```
cd data/raw
tar -xjf 20030228_easy_ham.tar.bz2
tar -xjf 20030228_easy_ham_2.tar.bz2
tar -xjf 20030228_hard_ham.tar.bz2
tar -xjf 20030228_spam.tar.bz2
tar -xjf 20030228_spam_2.tar.bz2
cd ../..
```
You should now have folders `data/raw/easy_ham`, `easy_ham_2`, `hard_ham`, `spam`, `spam_2`.

**4. Build the dataset, train, and test:**
```
python src/build_dataset.py
python src/train.py
python src/predict.py
pip install pytest
python -m pytest
```

**5. Start the website:**
```
python app.py
```
Open http://127.0.0.1:5000

## Deploy to GitHub and Render
1. Make sure `model/spam_pipeline.joblib` and `model/metrics.json` exist (they are committed on purpose).
2. Push to GitHub:
   ```
   git add .
   git commit -m "Spam classifier"
   git branch -M main
   git remote add origin https://github.com/<your-username>/AI-Spam-Classification.git
   git push -u origin main
   ```
3. On https://render.com choose **New + > Web Service**, connect the repository, and use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Health check path: `/health`
   (If you use the included `render.yaml`, these are filled in for you.)
4. The free tier sleeps when idle. The first request after a pause can take 30-60 seconds.

**Version rule:** the model is trained on your laptop and loaded on Render, so scikit-learn, numpy and pandas versions in `requirements.txt` must match your laptop. If you change versions, retrain and commit the new model.

## API
| Method | Route | Description |
|---|---|---|
| GET | `/` | Website |
| POST | `/predict` | Body `{"text": "..."}` returns label, spam_probability, confidence |
| GET | `/metrics` | Evaluation results |
| GET | `/health` | Health check |

## Technologies
Python, Pandas, NumPy, scikit-learn, Matplotlib, Flask, Gunicorn, HTML, CSS, JavaScript.
