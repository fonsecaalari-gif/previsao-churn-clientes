.PHONY: install train test app notebooks all

install:  ## Instala dependências e o pacote em modo editável
	pip install -r requirements.txt
	pip install -e .

train:  ## Treina, avalia e salva modelo, métricas e gráficos
	python -m churn.train

test:  ## Roda os testes automatizados
	pytest -q

app:  ## Abre o app interativo no navegador
	streamlit run app/streamlit_app.py

notebooks:  ## Reexecuta os notebooks de ponta a ponta
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

all: install test train notebooks
