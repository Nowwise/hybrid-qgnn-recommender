.PHONY: up down logs build launch

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
