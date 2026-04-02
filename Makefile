.PHONY: up down logs build launch prepare-movielens100k

prepare-movielens100k:
	python3 scripts/prepare_movielens100k.py --project-root .

up:
	docker compose up --build -d

launch:
	./launch.sh

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build
