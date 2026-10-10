dev:      ; docker compose up --build
test:     ; pytest -q
lint:     ; ruff check app tests
run:      ; uvicorn app.main:app --reload
