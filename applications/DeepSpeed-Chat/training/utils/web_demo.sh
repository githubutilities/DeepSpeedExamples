cd "$(dirname "${BASH_SOURCE}")"/..;

#lsof -n -i4TCP:8081 | awk '{print $2}' | xargs kill -9

python -m pip install -r ./requirements.txt

streamlit run --server.port=8081 ./web_demo.py

