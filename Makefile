PORT?=8050
USER := $(shell whoami)
USER_UID := $(shell id -u)
USER_GID := $(shell id -g)

clean:
	docker compose down --rmi local -v

build: clean
	USER=$(USER) docker compose build --progress=plain --no-cache

container:
	USER_UID=$(USER_UID) USER_GID=$(USER_GID) USER=$(USER) docker compose -p $(USER) up -d rd_template
	# -p (USER) adds `user-` at the beggining of the container name
