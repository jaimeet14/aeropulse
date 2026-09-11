.PHONY: install data train train-all test api frontend up down

PY ?= .venv/Scripts/python
ifeq ($(OS),)
PY = .venv/bin/python
endif

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt -e .

data:
	$(PY) scripts/download_data.py

train:
	$(PY) scripts/train.py --dataset FD001

train-all:
	$(PY) scripts/train.py --dataset all

test:
	$(PY) -m pytest

api:
	$(PY) -m uvicorn aeropredict.api.main:app --port 8010 --reload

frontend:
	cd frontend && npm install && npm run dev

up:
	docker compose up --build

down:
	docker compose down
