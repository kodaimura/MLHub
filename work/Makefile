SERVICE_NAME=julia-jupyter

.DEFAULT_GOAL := help

COMPOSE=docker compose

up:
	$(COMPOSE) up -d

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f $(SERVICE_NAME)

ps:
	$(COMPOSE) ps

url:
	@echo "Access Jupyter Notebook at: http://127.0.0.1:8888/lab?token=41790921b65fec995cfbb0357b6ab0a6536852d6e9d5942b"