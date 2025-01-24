DOCKER_COMPOSE = docker compose

.PHONY: up build down stop in log ps

up:
	$(DOCKER_COMPOSE) up -d

build:
	$(DOCKER_COMPOSE) build --no-cache

down:
	$(DOCKER_COMPOSE) down

stop:
	$(DOCKER_COMPOSE) stop

in:
	$(DOCKER_COMPOSE) exec julia bash

log:
	$(DOCKER_COMPOSE) logs -f

ps:
	$(DOCKER_COMPOSE) ps