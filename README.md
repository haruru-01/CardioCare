# CardioCare

Machine Learning final project for heart disease prediction.

## Project Structure

- notebooks/
  - EDA and preprocessing notebook

- src/
  - preprocessing.py
  - train.py
  - inference.py
  - monitor.py

- tests/
  - unit tests

- models/
  - trained model file

## Features

- Heart disease prediction
- Data preprocessing pipeline
- Model training and inference
- Drift detection using KS-test
- Unit testing
- Docker support
- GitHub Actions CI

## Run

Train model:

```bash
python src/train.py
```

Run inference:

```bash
python src/inference.py
```

Run monitoring:

```bash
python src/monitor.py
```

Run tests:

```bash
python -m unittest tests/test_pipeline.py
```

